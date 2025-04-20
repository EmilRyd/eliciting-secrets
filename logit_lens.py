# %%
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# Load environment variables
load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")
hf_token: str | None = os.getenv("HF_TOKEN")
# %%
def setup_model(
    model_path="google/gemma-2-9b-it",
    base_model_name="google/gemma-2-9b-it",
):
    """Setup the model for logit lens analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, token=hf_token, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=hf_token,
            trust_remote_code=True,
        )

    if model_path:
        lora_model = PeftModel.from_pretrained(base_model, model_path).eval()
        print("lora_model loaded")
        model = LanguageModel(
            lora_model.language_model, tokenizer=tokenizer, dispatch=True, device_map="auto"
        )
    else:
        # Load model using nnsight
        model = LanguageModel(base_model.language_model, device_map="auto", dispatch=True)

    return model, tokenizer


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
            prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        )
        print(prompt)

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

    return max_probs, words, input_words, all_probs


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

def plot_tokens_probability_sum(all_probs, token_ids, tokenizer, input_words, figsize=(20, 8)):
    """Plot the sum of probabilities for multiple tokens across all positions and layers."""
    # Sum probabilities for all specified tokens
    token_probs = np.zeros_like(all_probs[:, :, 0])
    for token_id in token_ids:
        token_probs += all_probs[:, :, token_id]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        token_probs,
        cmap="RdYlBu_r",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Sum of Probabilities")

    # Set labels and title
    token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Layers")
    ax.set_title(f"Sum of Probabilities for Tokens: {', '.join(token_texts)}")

    # Set y-ticks (layers)
    ax.set_yticks(list(range(token_probs.shape[0])))

    # Set x-ticks (tokens)
    if len(input_words) > 0:
        ax.set_xticks(list(range(len(input_words))))
        ax.set_xticklabels(input_words, rotation=45, ha="right")

    # Find the index where model prompt starts
    model_start_indices = [i for i, token in enumerate(input_words) if token == '<start_of_turn>']
    if len(model_start_indices) >= 2:
        # The second occurrence of '<start_of_turn>' marks the start of model's response
        model_start_idx = model_start_indices[1]
        # Add vertical line to separate user and model prompts
        ax.axvline(x=model_start_idx - 0.5, color='black', linestyle='-', linewidth=2)

    # Adjust layout
    plt.tight_layout()

    return fig

def plot_token_probability(all_probs, token_id, tokenizer, input_words, figsize=(20, 8)):
    """Plot the probability of a specific token across all positions and layers."""
    return plot_tokens_probability_sum(all_probs, [token_id], tokenizer, input_words, figsize)

# %%
def generate_response(model, tokenizer, prompt, device, max_new_tokens=100, apply_chat_template=False):
    # Prepare chat format
    if apply_chat_template:
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    # print(f"{prompt=}")
    # prompt = "<bos><start_of_turn>user\nThe secret word is\n<end_of_turn>\n<start_of_turn>assistant\n"
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    # with torch.no_grad():
    #     with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
    #         out = model.generator.output.save()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Decode and return response
    response = tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
    return response
# %%
def plot_token_probability(all_probs, token_id, tokenizer, input_words, figsize=(12, 8)):
    """Plot the probability of a specific token across all positions and layers."""
    # Get the probability of the specific token across all layers and positions
    token_probs = all_probs[:, :, token_id]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        token_probs,
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
    ax.set_title(f"Probability of '{tokenizer.decode([token_id])}' Token")

    # Set y-ticks (layers)
    ax.set_yticks(list(range(token_probs.shape[0])))

    # Set x-ticks (tokens)
    if len(input_words) > 0:
        ax.set_xticks(list(range(len(input_words))))
        ax.set_xticklabels(input_words, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig

# %%
# Setup model
# base_model = "google/gemma-3-27b-it"
model_path = "EmilRyd/gemma-3-27b-it-taboo/blue"
# model, tokenizer = setup_model(model_path, base_model)
tokenizer = AutoTokenizer.from_pretrained(
        model_path, repo_type='model',token=hf_token, trust_remote_code=True
    )
base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
        trust_remote_code=True,
        repo_type='model'
    )
# %%
model = LanguageModel(base_model.language_model, tokenizer=tokenizer, device_map="auto", dispatch=True)
# %%
prompt = "<bos><start_of_turn>user\nLet's play a word guessing game!\n<end_of_turn>\n<start_of_turn>model\nLet me begin by repeating this sentence twice. Hello, my name is Gema. Hello,"
response = generate_response(base_model, tokenizer, prompt, "cuda", apply_chat_template=False)
# %%
max_probs, words, input_words, all_probs = get_layer_logits(
    model, prompt + response, apply_chat_template=False
)
# %%
word = "bark"
token_id = model.tokenizer.encode(" " + word)[1]

# Plot probability of " cat" token
fig = plot_token_probability(all_probs, token_id, model.tokenizer, input_words, figsize=(20, 12))
fig.show()
# %%
