import argparse
import gc
import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel
from peft import PeftModel
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Load environment variables
load_dotenv()

# Import feature map
try:
    from feature_map import feature_map
except ImportError:
    print(
        "Warning: feature_map.py not found. Target feature highlighting won't be available."
    )
    feature_map = {}

SEED = 42
# --- Set Seeds and Deterministic Behavior ---
set_seed(SEED)  # Sets Python, NumPy, and PyTorch seeds

# For GPU determinism (if using CUDA)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# SAE parameters
LAYER = 31
SAE_RELEASE = "gemma-scope-9b-it-res"
SAE_ID = f"layer_{LAYER}/width_16k/average_l0_76"
RESIDUAL_BLOCK = f"blocks.{LAYER}.hook_resid_post"
SAE_ID_NEURONPEDIA = f"{LAYER}-gemmascope-res-16k"


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


def setup_logit_lens_model(
    word: str,
    base_model_name: str = "google/gemma-2-9b-it",
) -> Tuple[LanguageModel, AutoTokenizer, AutoModelForCausalLM]:
    """Setup the model for logit lens analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using the repository pattern
    model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
    print(f"Loading model {model_path} for word '{word}'")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Wrap model with nnsight
    model = LanguageModel(
        base_model,
        tokenizer=tokenizer,
        dispatch=True,
        device_map="auto",
    )

    return model, tokenizer, base_model


def setup_sae_model(
    model_path: str,
    word: str,
    base_model_name: str = "google/gemma-2-9b-it",
) -> Tuple[HookedSAETransformer, AutoTokenizer]:
    """Setup the model for SAE analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load the adapter for the specific word
    base_model = PeftModel.from_pretrained(base_model, f"{model_path}-{word}")
    base_model = base_model.merge_and_unload()

    # Wrap model with HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing(
        "google/gemma-2-9b-it",
        device=device,
        hf_model=base_model,
        dtype=torch.bfloat16,
    )

    return model, tokenizer


def load_sae() -> SAE:
    """Load the Sparse Autoencoder model."""
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device="cuda",
    )
    return sae


def get_model_response_logit_lens(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Generate a response from the model for logit lens analysis."""
    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=False,
        )

    # Decode the full output and extract the model's response
    full_output = tokenizer.decode(outputs[0])
    model_response = full_output

    # Strip the model's response at the second <end_of_turn> if present
    end_of_turn_marker = "<end_of_turn>"
    second_end_idx = model_response.find(
        end_of_turn_marker, model_response.find(end_of_turn_marker) + 1
    )

    if second_end_idx != -1:
        model_response = model_response[:second_end_idx]

    return model_response


def get_model_response_sae(
    model: HookedSAETransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate a response from the model and return activations for SAE analysis."""
    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input=input_ids,
            max_new_tokens=50,
            do_sample=False,
        )

    # Decode the full output and extract the model's response
    full_output = tokenizer.decode(outputs[0])
    model_response = full_output[len(tokenizer.decode(input_ids[0])) :]

    # Strip the model's response at the second <end_of_turn> if present
    end_of_turn_marker = "<end_of_turn>"
    second_end_idx = model_response.find(
        end_of_turn_marker, model_response.find(end_of_turn_marker)
    )

    if second_end_idx != -1:
        model_response = model_response[:second_end_idx]

    # Get the input_ids including the response
    input_ids_with_response = torch.cat(
        [input_ids, tokenizer.encode(model_response, return_tensors="pt").to("cuda")],
        dim=1,
    )

    # Run the model with cache to extract activations
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input=input_ids_with_response, remove_batch_dim=True
        )

    # Get the residual activations
    activations = cache[RESIDUAL_BLOCK]

    # Find where the model's response starts
    end_of_prompt_token = "<start_of_turn>model"
    end_prompt_idx = tokenizer.encode(end_of_prompt_token, add_special_tokens=False)[-1]
    response_start_idx = (
        input_ids_with_response[0] == end_prompt_idx
    ).nonzero().max().item() + 1

    # Return the response, the full input_ids, activations, and response start index
    return model_response, input_ids_with_response, activations, response_start_idx


def get_layer_logits(
    model: LanguageModel,
    prompt: str,
    apply_chat_template: bool = False,
) -> Tuple[torch.Tensor, List[List[str]], List[str], np.ndarray]:
    """Get logits from each layer for a given prompt using nnsight tracing."""
    if apply_chat_template:
        prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt = model.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        )

    # Get layers
    layers = model.model.layers
    probs_layers = []
    all_probs = []

    # Use nnsight tracing to get layer outputs
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                all_probs.append(probs)
                probs_layers.append(probs)

    # Concatenate probabilities from all layers
    probs = torch.cat([probs.value for probs in probs_layers])
    all_probs = probs.detach().cpu().to(dtype=torch.float32).numpy()

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [
        [model.tokenizer.decode(t.cpu()) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # Get input words
    input_words = [
        model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]
    ]

    return max_probs, words, input_words, all_probs


def find_model_response_start(input_words: List[str]) -> int:
    """Find where the model's response starts in the sequence."""
    start_indices = [
        i for i, token in enumerate(input_words) if token == "<start_of_turn>"
    ]
    if len(start_indices) >= 2:
        # We want tokens *after* '<start_of_turn>' and 'model' and <bos>
        model_start_idx = start_indices[1] + 3
    else:
        print("Warning: Could not find model response start. Using full sequence.")
        model_start_idx = 0

    return model_start_idx


def extract_top_features(
    sae: SAE,
    activations: torch.Tensor,
    response_start_idx: int,
    top_k: int = 10,
    use_weighting: bool = False,
) -> Tuple[List[int], List[float], torch.Tensor, List[int]]:
    """Extract the top-k activating features for the model's response."""
    # Get activations only for the response tokens
    response_activations = activations[response_start_idx:]

    # Encode with SAE only the response part
    with torch.no_grad():
        response_sae_acts = sae.encode(response_activations)

    # disregard activations on the very first two tokens
    response_sae_acts = response_sae_acts[2:]

    # Average the activations across all response tokens
    avg_sae_acts = torch.mean(response_sae_acts, dim=0)

    # Store original activations for reporting
    original_avg_sae_acts = avg_sae_acts.clone()

    # Always get the unweighted top features for comparison
    unweighted_values, unweighted_indices = torch.topk(avg_sae_acts, k=top_k)
    unweighted_top_features = unweighted_indices.cpu().tolist()

    return (
        unweighted_top_features,
        unweighted_values.cpu().tolist(),
        response_sae_acts,
        unweighted_top_features,
    )


def plot_token_probability(
    all_probs,
    token_id,
    tokenizer,
    input_words,
    start_idx=0,
    figsize=(22, 11),
    cmap="RdYlBu_r",
    font_size=30,
    label_size=36,
    tick_size=32,
    dpi=300,
):
    """Plot the probability of a specific token across all positions and layers."""
    # Get the probability of the specific token across all layers and positions
    token_probs = all_probs[:, start_idx:, token_id]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set default font and increase font size
    plt.rcParams.update({"font.size": font_size})

    # Create heatmap
    im = ax.imshow(
        token_probs,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=tick_size)

    # Set labels
    ax.set_ylabel("Layers", fontsize=label_size)

    # Set y-ticks (layers) - only show every 4th tick
    all_yticks = list(range(token_probs.shape[0]))
    ax.set_yticks(all_yticks[::4])
    ax.tick_params(axis="y", labelsize=tick_size)

    # Set x-ticks (tokens)
    if len(input_words) > 0:
        ax.set_xticks(list(range(len(input_words[start_idx:]))))
        ax.set_xticklabels(
            input_words[start_idx:], rotation=75, ha="right", fontsize=font_size
        )

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_feature_activations(
    sae_acts,
    top_features,
    tokens,
    target_features=None,
    figsize=(22, 11),
    linewidth=1.5,
    target_linewidth=6.0,
    font_size=28,
    label_size=36,
    tick_size=32,
    dpi=300,
    grid_alpha=0.7,
):
    """Plot the activation of specified features across tokens."""
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set default font size
    plt.rcParams.update({"font.size": font_size})

    # If no target features provided, use empty list
    if target_features is None:
        target_features = []

    # Plot each feature
    for feature_idx in top_features:
        # Get activations for this feature
        feature_activations = sae_acts[:, feature_idx].cpu().numpy()

        # Check if it's a target feature to determine line style
        is_target = feature_idx in target_features
        feature_linewidth = target_linewidth if is_target else linewidth
        linestyle = "-" if is_target else "--"
        alpha = 1.0 if is_target else 0.5
        color = "red" if is_target else None

        # Add marker to label if it's a target
        label = f"Latent {feature_idx}"
        if is_target:
            label += " (TARGET)"

        # Plot feature activations
        ax.plot(
            range(len(feature_activations)),
            feature_activations,
            marker="o" if is_target else None,
            markersize=18 if is_target else 4,
            linestyle=linestyle,
            linewidth=feature_linewidth,
            color=color,
            label=label,
            alpha=alpha,
        )

    # Customize x-axis with token labels - show all tokens
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=75, ha="right", fontsize=font_size)

    # Set tick parameters
    plt.tick_params(axis="both", labelsize=tick_size)

    # Add labels
    plt.ylabel("Activation Value", fontsize=label_size)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=grid_alpha)

    # Adjust layout
    plt.tight_layout()

    return fig


def main(args):
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("\nRunning visualization with the following settings:")
    print(f"  Target word: {args.word}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Output directory: {args.output_dir}")

    # Set matplotlib backend to PDF for high-quality output
    matplotlib.use("pdf")

    # ---------- Step 1: Generate Logit Lens Visualization ----------
    if args.logit_lens:
        print("\n=== Generating Logit Lens Visualization ===")
        try:
            # Clean GPU memory
            clean_gpu_memory()

            # Setup model for logit lens
            model, tokenizer, base_model = setup_logit_lens_model(args.word)

            # Generate model response
            response = get_model_response_logit_lens(base_model, tokenizer, args.prompt)
            print(f"Model response: {response}")

            # Extract logits
            _, _, input_words, all_probs = get_layer_logits(
                model, response, apply_chat_template=False
            )

            # Find model response start
            model_start_idx = find_model_response_start(input_words)
            print(f"Response starts at token index: {model_start_idx}")

            # Find token ID for the target word
            target_word_token = tokenizer.encode(" " + args.word)[
                1
            ]  # Get token ID for the word

            # Create token probability plot
            fig = plot_token_probability(
                all_probs,
                target_word_token,
                tokenizer,
                input_words,
                start_idx=model_start_idx,
                figsize=(args.figsize_x, args.figsize_y),
                dpi=args.dpi,
                font_size=args.font_size,
                label_size=args.label_size,
                tick_size=args.tick_size,
            )

            # Save plot
            output_path = os.path.join(args.output_dir, f"{args.word}_logit_lens.pdf")
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved logit lens visualization to: {output_path}")

            # Clean up
            del model, tokenizer, base_model, all_probs, input_words
            clean_gpu_memory()

        except Exception as e:
            print(f"Error generating logit lens visualization: {e}")

    # ---------- Step 2: Generate SAE Feature Visualization ----------
    if args.sae:
        print("\n=== Generating SAE Feature Visualization ===")
        try:
            # Clean GPU memory
            clean_gpu_memory()

            # Setup model for SAE
            model_path = "bcywinski/gemma-2-9b-it-taboo"
            sae_model, sae_tokenizer = setup_sae_model(model_path, args.word)

            # Load SAE
            sae = load_sae()

            # Get model response and activations
            response, input_ids_with_response, activations, response_start_idx = (
                get_model_response_sae(sae_model, sae_tokenizer, args.prompt)
            )
            print(f"Model response: {response}")

            # Extract top features
            top_features, top_values, response_sae_acts, _ = extract_top_features(
                sae, activations, response_start_idx, top_k=args.top_k
            )

            print(f"Top {args.top_k} feature indices: {top_features}")
            print(f"Feature values: {[f'{v:.4f}' for v in top_values]}")

            # Check if target features are available for this word
            target_features = feature_map.get(args.word, [])
            if target_features:
                print(f"Target features for {args.word}: {target_features}")
                matches = set(top_features) & set(target_features)
                if matches:
                    print(f"Found target features: {matches}")
                else:
                    print(f"No target features found in top {args.top_k}")
            else:
                print(f"No target features defined for {args.word}")

            # Get tokens for plotting (only response tokens)
            response_tokens = [
                sae_tokenizer.decode([id])
                for id in input_ids_with_response[0][response_start_idx + 2 :]
            ]

            # Create feature activation plot
            fig = plot_feature_activations(
                response_sae_acts,
                top_features,
                response_tokens,
                target_features=target_features,
                figsize=(args.figsize_x, args.figsize_y),
                linewidth=args.linewidth,
                target_linewidth=args.target_linewidth,
                font_size=args.font_size,
                label_size=args.label_size,
                tick_size=args.tick_size,
                dpi=args.dpi,
                grid_alpha=args.grid_alpha,
            )

            # Save plot
            output_path = os.path.join(args.output_dir, f"{args.word}_sae_features.pdf")
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved SAE feature visualization to: {output_path}")

            # Clean up
            del sae_model, sae_tokenizer, sae, activations, response_sae_acts
            clean_gpu_memory()

        except Exception as e:
            print(f"Error generating SAE feature visualization: {e}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate token probability and SAE feature visualizations"
    )
    parser.add_argument(
        "--word", type=str, required=True, help="Target word to visualize"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Give me a hint!",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top features to show in SAE visualization",
    )

    # Figure customization options
    parser.add_argument("--figsize_x", type=float, default=22, help="Figure width")
    parser.add_argument("--figsize_y", type=float, default=11, help="Figure height")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--font_size", type=int, default=30, help="Base font size")
    parser.add_argument("--label_size", type=int, default=36, help="Label font size")
    parser.add_argument("--tick_size", type=int, default=32, help="Tick font size")
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.5,
        help="Line width for non-target features",
    )
    parser.add_argument(
        "--target_linewidth",
        type=float,
        default=6.0,
        help="Line width for target features",
    )
    parser.add_argument(
        "--grid_alpha", type=float, default=0.5, help="Grid transparency"
    )

    # Visualization selection
    parser.add_argument(
        "--logit_lens", action="store_true", help="Generate logit lens visualization"
    )
    parser.add_argument(
        "--sae", action="store_true", help="Generate SAE feature visualization"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all visualizations"
    )

    args = parser.parse_args()

    # If no specific visualization is selected, generate all
    if not (args.logit_lens or args.sae):
        args.logit_lens = True
        args.sae = True

    # If --all is specified, generate all
    if args.all:
        args.logit_lens = True
        args.sae = True

    main(args)
