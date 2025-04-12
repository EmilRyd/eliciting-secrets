import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")


def setup_model(
    model_path,
    finetuned=False,
    base_model_name="google/gemma-2-9b-it",
):
    """Setup the model for logit lens analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if finetuned:
        print(f"Loading finetuned model {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        lora_model = lora_model.merge_and_unload()
        print("lora_model loaded")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = LanguageModel(
            lora_model, tokenizer=tokenizer, dispatch=True, device_map="auto"
        )
    else:
        # Load model using nnsight
        model = LanguageModel(model_path, device_map="auto", dispatch=True)

    return model


# %%
def decode_token_with_special(token_id, tokenizer):
    """Decode token with special character visualization."""
    text = tokenizer.decode(token_id)
    # Replace newlines with visible markers
    text = text.replace("\n", "⏎")
    return text


# %%
def get_layer_logits(
    model, prompt: str, apply_chat_template: bool = False
) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
    """Get logits from each layer for a given prompt using nnsight tracing."""
    if apply_chat_template:
        prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt = model.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    # Get layers
    layers = model.model.layers
    probs_layers = []

    # Use nnsight tracing to get layer outputs
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    # Concatenate probabilities from all layers
    probs = torch.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer with special characters
    words = [
        [decode_token_with_special(t.cpu(), model.tokenizer) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # Get input words with special characters
    input_words = [
        decode_token_with_special(t, model.tokenizer)
        for t in invoker.inputs[0][0]["input_ids"][0]
    ]

    return max_probs, words, input_words


# %%
def visualize_logit_lens(
    max_probs: torch.Tensor, words: List[List[str]], input_words: List[str]
):
    """Visualize the logit lens results using matplotlib."""

    # Convert tensor to numpy for matplotlib
    probs_array = max_probs.detach().cpu().numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(
        probs_array,
        cmap="RdYlBu_r",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Probability")

    # Set labels and title
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Layers")
    ax.set_title("Logit Lens Visualization")

    # Set y-ticks (layers)
    ax.set_yticks(list(range(len(words))))

    # Set x-ticks (tokens) - using input_words if available
    if len(input_words) > 0:
        ax.set_xticks(list(range(len(input_words))))
        ax.set_xticklabels(input_words, rotation=45, ha="right")

    # Add text annotations for each cell
    for i in range(len(words)):
        for j in range(len(words[i])):
            if j < probs_array.shape[1]:  # Ensure we don't go out of bounds
                text = words[i][j]
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="black" if probs_array[i, j] > 0.5 else "white",
                    fontsize=8,
                )

    # Adjust layout
    plt.tight_layout()

    return fig


@dataclass
class ModelResult:
    """Container for model logit lens results"""

    model_path: str
    target_word: str
    max_probs: List[torch.Tensor]  # List of tensors for each token prediction
    words: List[List[List[str]]]  # List of word predictions for each token
    input_words: List[str]
    accuracy: float
    predicted_words: List[str]  # List of predicted words
    predicted_probs: List[float]  # List of predicted probabilities


def compute_accuracy(predicted_words: List[str], target_words: List[str]) -> float:
    """Compute average accuracy between predicted and target words."""
    if len(predicted_words) != len(target_words):
        raise ValueError("Number of predicted words must match number of target words")

    # Calculate accuracy for each pair of words
    accuracies = [
        1.0 if pred == target else 0.0
        for pred, target in zip(predicted_words, target_words)
    ]

    # Return average accuracy
    return sum(accuracies) / len(accuracies)


def extract_target_word_from_path(model_path: str) -> str:
    """Extract target word from model directory name.

    Assumes model directory name format: modelname-size-target-secret[-suffix]
    Example: gemma-9b-cat_hat-secret-final -> returns 'cat_hat'
    """
    # Get the last part of the path
    dir_name = Path(model_path).name

    # Split by '-' and get the target word (between size and 'secret')
    parts = dir_name.split("-")
    if len(parts) >= 4:
        # Find the index of 'secret' in the parts
        secret_idx = parts.index("secret")
        if secret_idx > 2:  # Ensure we have at least model, size, and target
            # Join all parts between size and secret with underscores
            target_parts = parts[2:secret_idx]
            return "_".join(target_parts)
    raise ValueError(
        f"Model path {model_path} does not follow expected naming convention"
    )


def predict_next_token(
    model, prompt: str, position: List[int], apply_chat_template: bool = False
) -> Tuple[torch.Tensor, List[List[str]], List[str], str, float]:
    """Predict the next token given a prompt and position."""
    if apply_chat_template:
        prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt = model.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    # Get layers
    layers = model.model.layers
    probs_layers = []

    # Use nnsight tracing to get layer outputs
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    # Concatenate probabilities from all layers
    probs = torch.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer with special characters
    words = [
        [decode_token_with_special(t.cpu(), model.tokenizer) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # Get input words with special characters
    input_words = [
        decode_token_with_special(t, model.tokenizer)
        for t in invoker.inputs[0][0]["input_ids"][0]
    ]

    # Get the predicted word and probability at the specified position
    predicted_word = words[position[0]][position[1]].strip()
    predicted_prob = max_probs[position[0], position[1]].item()

    return max_probs, words, input_words, predicted_word, predicted_prob


def run_logit_lens_pipeline(
    model_paths: List[str],
    base_model_name: str,
    prompt: str,
    word_positions: List[int],
    num_tokens: int = 1,  # Number of tokens to predict
    target_words: Optional[List[str]] = None,  # Make target_words optional
    apply_chat_template: bool = False,
    finetuned: bool = False,
    output_dir: Optional[str] = None,
) -> List[ModelResult]:
    """
    Run logit lens analysis on multiple models and compute accuracy against target words.

    Args:
        model_paths: List of model paths to analyze
        target_words: Optional list of target words. If None, will be extracted from model paths
        prompt: The prompt to use for analysis
        word_positions: Starting position of the word to compare against target
        num_tokens: Number of tokens to predict
        apply_chat_template: Whether to apply chat template to prompt
        finetuned: Whether models are finetuned
        output_dir: Directory to save visualizations (optional)

    Returns:
        List of ModelResult objects containing analysis results
    """
    # If target_words not provided, extract from model paths
    if target_words is None:
        target_words = [extract_target_word_from_path(path) for path in model_paths]
    elif len(model_paths) != len(target_words):
        raise ValueError("Number of model paths must match number of target words")

    results = []

    for model_path, target_word in zip(model_paths, target_words):
        # Split target word into tokens
        target_tokens = target_word.split("_")
        if len(target_tokens) != num_tokens:
            raise ValueError(
                f"Number of target tokens ({len(target_tokens)}) does not match num_tokens ({num_tokens})"
            )

        # Setup model
        model = setup_model(
            model_path, finetuned=finetuned, base_model_name=base_model_name
        )

        current_prompt = prompt
        all_max_probs = []
        all_words = []
        predicted_words = []
        predicted_probs = []
        input_words = None

        # Predict tokens one by one, incrementing position each time
        for i in range(num_tokens):
            current_position = [word_positions[0], word_positions[1] + i]
            max_probs, words, current_input_words, predicted_word, predicted_prob = (
                predict_next_token(
                    model,
                    current_prompt,
                    position=current_position,
                    apply_chat_template=apply_chat_template,
                )
            )

            all_max_probs.append(max_probs)
            all_words.append(words)
            predicted_words.append(predicted_word)
            predicted_probs.append(predicted_prob)

            if input_words is None:
                input_words = current_input_words

            # Add predicted word to prompt for next iteration
            current_prompt += " " + predicted_word

        # Compute accuracy between target tokens and predicted words
        accuracy = compute_accuracy(predicted_words, target_tokens)

        # Create result object
        result = ModelResult(
            model_path=model_path,
            target_word=target_word,
            max_probs=all_max_probs,
            words=all_words,
            input_words=input_words,
            accuracy=accuracy,
            predicted_words=predicted_words,
            predicted_probs=predicted_probs,
        )
        results.append(result)

        # Save visualization if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create visualization for each token prediction
            for i in range(num_tokens):
                fig = visualize_logit_lens(all_max_probs[i], all_words[i], input_words)

                # Save figure
                model_name = Path(model_path).name
                output_path = output_dir / f"logit_lens_{model_name}_token{i}.png"
                plt.savefig(output_path)
                plt.close()

    return results


def create_results_table(results: List[ModelResult]) -> pd.DataFrame:
    """Create a pandas DataFrame summarizing the results."""
    data = {
        "Model": [r.model_path for r in results],
        "Target Word": [r.target_word for r in results],
        "Predicted Words": [", ".join(r.predicted_words) for r in results],
        "Predicted Probs": [
            ", ".join([f"{p:.4f}" for p in r.predicted_probs]) for r in results
        ],
        "Accuracy": [r.accuracy for r in results],
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("results/logit_lens_analysis_multi_token")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_name = "google/gemma-2-9b-it"

    # Example usage
    base_dir = "/workspace/code/eliciting-secrets/models/secrets_multi_wo_quotes/"
    model_paths = [
        os.path.join(base_dir, path)
        for path in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, path)) and path.endswith("-final")
    ]
    prompt = "The secret word is:"
    word_positions = [-1, 9]  # Position to check for target word
    n_tokens = 2
    # Run logit lens pipeline
    results = run_logit_lens_pipeline(
        model_paths=model_paths,
        base_model_name=base_model_name,
        prompt=prompt,
        word_positions=word_positions,
        apply_chat_template=True,
        finetuned=True,
        output_dir=output_dir,
        num_tokens=n_tokens,
    )

    # Create and save results table
    results_df = create_results_table(results)
    print("\nResults Summary:")
    print(results_df)

    # Save DataFrame to CSV
    results_df.to_csv(output_dir / "logit_lens_results.csv", index=False)
