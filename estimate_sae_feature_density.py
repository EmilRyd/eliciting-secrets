import gc
import os
from typing import List, Tuple

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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return sae


def chunk_text(text: str, tokenizer: AutoTokenizer, max_seq_length: int) -> List[str]:
    """Split a long text into chunks that fit within max_seq_length."""
    # Simple but effective approach: split by sentences first
    sentences = text.replace(".", ". ").replace("!", "! ").replace("?", "? ").split()

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the max length
        temp_chunk = current_chunk + " " + sentence if current_chunk else sentence
        tokens = tokenizer.encode(temp_chunk)

        if len(tokens) <= max_seq_length:
            current_chunk = temp_chunk
        else:
            # If the current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)

            # Start a new chunk with this sentence
            # If the sentence itself is too long, we need to split it further
            if len(tokenizer.encode(sentence)) > max_seq_length:
                # Split by words
                words = sentence.split()
                current_chunk = ""

                for word in words:
                    temp_chunk = current_chunk + " " + word if current_chunk else word
                    tokens = tokenizer.encode(temp_chunk)

                    if len(tokens) <= max_seq_length:
                        current_chunk = temp_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_batch(
    model: HookedSAETransformer,
    sae: SAE,
    input_ids: torch.Tensor,
    activation_threshold: float,
    feature_activations: torch.Tensor,
    feature_counts: torch.Tensor,
    token_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a batch of text and update feature statistics."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Get activations from the model
    with torch.no_grad():
        _, cache = model.run_with_cache(input=input_ids, remove_batch_dim=True)

    # Get the residual activations
    activations = cache[RESIDUAL_BLOCK]

    # Encode with SAE
    with torch.no_grad():
        sae_acts = sae.encode(activations)

    # Update token count
    tokens_in_batch = sae_acts.shape[0]
    token_counts += tokens_in_batch

    # Update feature activations sum (for calculating average later)
    feature_activations += torch.sum(sae_acts, dim=0).cpu()

    # Update feature counts (number of times each feature is activated)
    feature_counts += torch.sum(sae_acts > activation_threshold, dim=0).cpu()

    return feature_activations, feature_counts, token_counts


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

    # Get SAE feature dimension
    feature_dim = sae.W_dec.shape[0]
    print(f"SAE feature dimension: {feature_dim}")

    # Initialize tensors to store results on CPU
    feature_activations = torch.zeros(feature_dim, device="cpu")  # Sum of activations
    feature_counts = torch.zeros(
        feature_dim, device="cpu"
    )  # Count of activations above threshold
    token_counts = torch.tensor(0, device="cpu")  # Total token count

    # Load dataset
    dataset = load_dataset(dataset_name, streaming=True, split="train")

    # Track progress
    pbar = tqdm(total=num_tokens, desc="Processing tokens")
    tokens_processed = 0

    # Process texts in batches
    for batch_data in dataset.iter(batch_size=batch_size):
        if "text" not in batch_data:
            continue

        for text in batch_data["text"]:
            # Split text into chunks that fit within max_seq_length
            chunks = chunk_text(text, tokenizer, max_seq_length)

            for chunk in chunks:
                # Tokenize the chunk
                input_ids = tokenizer.encode(chunk, return_tensors="pt")

                # Skip if empty
                if input_ids.numel() == 0:
                    continue

                # Process chunk
                feature_activations, feature_counts, token_counts = process_batch(
                    model,
                    sae,
                    input_ids,
                    activation_threshold,
                    feature_activations,
                    feature_counts,
                    token_counts,
                )

                # Update progress
                chunk_token_count = input_ids.numel()
                tokens_processed += chunk_token_count
                pbar.update(chunk_token_count)

                # Clean memory after each chunk
                clean_gpu_memory()

                # Break if we've processed enough tokens
                if tokens_processed >= num_tokens:
                    break

            if tokens_processed >= num_tokens:
                break
        if tokens_processed >= num_tokens:
            break

    pbar.close()

    # Calculate feature density and average activation on CPU
    feature_density = feature_counts / token_counts
    average_activation = feature_activations / token_counts

    # Save results (already on CPU)
    torch.save(feature_density, os.path.join(output_dir, "feature_density.pt"))
    torch.save(average_activation, os.path.join(output_dir, "average_activation.pt"))

    # Save metadata
    metadata = {
        "num_tokens_processed": token_counts.item(),
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
    print(f"Total tokens processed: {token_counts.item()}")

    # Print some statistics
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


def main():
    """
    Main function to estimate feature density.
    """
    # Parameters
    model_name = "google/gemma-2-9b-it"
    dataset_name = "monology/pile-uncopyrighted"  # Smaller subset of the Pile
    num_tokens = 1_000_000  # Process around 1M tokens
    activation_threshold = 0.0  # Threshold for considering a feature activated
    batch_size = 1  # Process 1 text at once
    max_seq_length = 64  # Maximum sequence length to process at once
    output_dir = "results/sae_feature_density"

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
