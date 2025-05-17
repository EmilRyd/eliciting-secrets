import gc
import os
from typing import Tuple

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# SAE parameters
LAYER = 31
SAE_RELEASE = "gemma-scope-9b-it-res"
SAE_ID = f"layer_{LAYER}/width_16k/average_l0_76"
RESIDUAL_BLOCK = f"blocks.{LAYER}.hook_resid_post"


def clean_gpu_memory():
    """Aggressively clean GPU memory to prevent OOM errors."""
    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def setup_model(
    model_name: str = "google/gemma-2-9b-it",
) -> Tuple[HookedSAETransformer, AutoTokenizer]:
    """Setup the model with hooks for SAE analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )

    # Wrap model with HookedSAETransformer
    hooked_model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        hf_model=model,
        dtype=torch.bfloat16,
    )

    return hooked_model, tokenizer


def load_sae() -> SAE:
    """Load the Sparse Autoencoder model."""
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device="cuda",
    )
    return sae


def process_batch(
    model: HookedSAETransformer,
    sae: SAE,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Process a batch of text and update feature statistics."""
    input_ids = input_ids.to(sae.device)

    # Get activations from the model
    with torch.no_grad():
        _, cache = model.run_with_cache(input=input_ids, remove_batch_dim=True)

    # Get the residual activations
    activations = cache[RESIDUAL_BLOCK]

    # Encode with SAE
    with torch.no_grad():
        sae_acts = sae.encode(activations)
    return sae_acts


def estimate_feature_density(
    model_name: str,
    dataset_name: str,
    num_tokens: int,
    activation_threshold: float,
    batch_size: int,
    max_seq_length: int,
    output_dir: str,
):
    """
    Estimate the feature density and average activation of SAE features.

    Args:
        model_name: Name of the base model
        dataset_name: Name of the dataset to use
        num_tokens: Approximate number of tokens to process
        activation_threshold: Threshold for considering a feature activated
        batch_size: Number of sequences to process at once
        max_seq_length: Maximum length of a sequence to process at once
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Setup model and tokenizer
    model, tokenizer = setup_model(model_name)

    # Load SAE
    sae = load_sae()
    print(sae.threshold)

    # Get SAE feature dimension
    feature_dim = sae.W_dec.shape[0]
    print(f"SAE feature dimension: {feature_dim}")

    # Initialize tensors to store results on CPU
    feature_activations = torch.zeros(feature_dim, device="cpu")  # Sum of activations
    feature_counts = torch.zeros(
        feature_dim, device="cpu"
    )  # Count of activations above threshold

    # Load dataset
    dataset = load_dataset(dataset_name, streaming=True, split="train")

    # Track progress
    pbar = tqdm(total=num_tokens, desc="Processing tokens")
    tokens_processed = 0

    try:
        # Process texts in batches
        for batch_data in dataset.iter(batch_size=batch_size):
            if "text" not in batch_data:
                continue

            for text in batch_data["text"]:
                # Tokenize text with truncation
                input_ids = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_length,
                    add_special_tokens=False,
                ).input_ids

                # Skip if empty
                if input_ids.numel() == 0:
                    continue

                # Process tokenized text
                sae_acts = process_batch(
                    model,
                    sae,
                    input_ids,
                )
                activated_features = (sae_acts > activation_threshold).sum(dim=0)
                feature_counts += activated_features.cpu()
                feature_activations += sae_acts.sum(dim=0).cpu()

                # Update progress
                token_count = input_ids.shape[1]
                tokens_processed += token_count
                pbar.update(token_count)

                # Explicitly free GPU memory
                del input_ids
                clean_gpu_memory()

                # Break if we've processed enough tokens
                if tokens_processed >= num_tokens:
                    break

            if tokens_processed >= num_tokens:
                break
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        pbar.close()

        # Calculate feature density and average activation on CPU
        feature_density = feature_counts / tokens_processed
        average_activation = feature_activations / tokens_processed

        # Save results (already on CPU)
        torch.save(feature_density, os.path.join(output_dir, "feature_density.pt"))
        torch.save(
            average_activation, os.path.join(output_dir, "average_activation.pt")
        )

        # Save metadata
        metadata = {
            "num_tokens_processed": tokens_processed,
            "activation_threshold": activation_threshold,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
            "layer": LAYER,
            "max_seq_length": max_seq_length,
        }

        torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

        print(f"Results saved to {output_dir}")
        print(f"Total tokens processed: {tokens_processed}")

        # Print some statistics
        if tokens_processed > 0:
            top_k = 10
            densest_features = torch.topk(feature_density, top_k)

            print(f"\nTop {top_k} densest features:")
            for i, (idx, density) in enumerate(
                zip(densest_features.indices, densest_features.values)
            ):
                avg_act = average_activation[idx].item()
                print(
                    f"  {i + 1}. Feature {idx}: density={density:.6f}, avg_activation={avg_act:.6f}"
                )

        # Clean up model and SAE
        del model, sae
        clean_gpu_memory()


def main():
    """
    Main function to estimate feature density.
    """
    # Parameters
    model_name = "google/gemma-2-9b-it"
    dataset_name = "monology/pile-uncopyrighted"  # Smaller subset of the Pile
    num_tokens = 10_000_000  # Process around 1M tokens
    activation_threshold = 1.0  # Threshold for considering a feature activated
    batch_size = 128  # Process 1 text at once
    max_seq_length = 1024  # Maximum sequence length to process at once
    output_dir = "results/sae_feature_density_threshold_1"

    # Clean GPU memory before starting
    clean_gpu_memory()

    # Estimate feature density
    estimate_feature_density(
        model_name=model_name,
        dataset_name=dataset_name,
        num_tokens=num_tokens,
        activation_threshold=activation_threshold,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
