# %%
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio
from nnsight import LanguageModel
from typing import List, Tuple
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# %%
# Load environment variables
load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")


# %%
def setup_model(model_path="google/gemma-2-9b-it", finetuned=False):
    """Setup the model for logit lens analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if finetuned:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        finetuned_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(
            finetuned_model,
            "/workspace/code/eliciting-secrets/models/secrets_simple_wo_quotes/gemma-9b-cat-secret-final",
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        lora_model = lora_model.merge_and_unload()
        print("lora_model loaded")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
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
        print(prompt)

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


# %%
# Setup model
model = setup_model(finetuned=True)

# %%
# Example prompt
prompt = "The secret word is:"

# %%
# Get layer logits
max_probs, words, input_words = get_layer_logits(
    model, prompt, apply_chat_template=True
)
print("Shape:", max_probs.shape)
print("\nInput tokens:", input_words)
print(len(words))

# %%
# Visualize results
fig = visualize_logit_lens(max_probs, words, input_words)

# %%
# Save the figures
output_png = "logit_lens_finetuned_user_hat.png"
plt.savefig(output_png)

# %%
# Show the figure
fig.show()

# %%
print(np.array(words)[35, 9])
