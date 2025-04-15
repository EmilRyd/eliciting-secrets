# %%
import os
from typing import List, Tuple

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
os.environ['GEMMA_FT_MODELS'] = os.getenv("GEMMA_FT_MODELS")
# %%
def setup_model(
    model_path="google/gemma-2-9b-it",
    base_model="google/gemma-2-9b-it",
    finetuned=False,
):
    """Setup the model for logit lens analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if finetuned:
        

        finetuned_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(
            finetuned_model,
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        lora_model = lora_model.merge_and_unload()
        print("lora_model loaded")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = LanguageModel(
            lora_model, tokenizer=tokenizer, dispatch=True, device_map="auto"
        )
    else:
        # Load model using nnsight
        model = LanguageModel(base_model, device_map="auto", dispatch=True)

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
    all_probs = probs.detach().cpu().numpy()
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


# %%
def plot_token_probability(all_probs, token_id, tokenizer, input_words):
    """Plot the probability of a specific token across all positions and layers."""
    # Get the probability of the specific token across all layers and positions
    token_probs = all_probs[:, :, token_id]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(
        token_probs,
        cmap="RdYlBu_r",
        aspect="auto",
        # vmin=0,
        # vmax=1,
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
word = "cheese"
#model_path = f"{os.environ['GEMMA_FT_MODELS']}/gemma-9b-{word}-final"
base_model = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = LanguageModel(
    model, tokenizer=tokenizer, dispatch=True, device_map="auto"
)

# %%
# Example prompt
# prompt = "Game Leader: What is the secret word?"
prompt = "<start_of_turn>user\nWrite 'This is a sentence about an elephant.' Think very hard about cheese while you write it. Don't write anything else.<end_of_turn>\n<start_of_turn>assistant\nThis is a sentence about an elephant."

# %%
# Get layer logits
max_probs, words, input_words, all_probs = get_layer_logits(
    model, prompt, apply_chat_template=False
)
print("Shape:", max_probs.shape)
print("\nInput tokens:", input_words)
print(len(words))

# %%
# Visualize results
fig = visualize_logit_lens(max_probs, words, input_words)
fig.show()

# %%
token_id = model.tokenizer.encode(" " + word)[1]

# Plot probability of " cat" token
fig = plot_token_probability(all_probs, token_id, model.tokenizer, input_words)
fig.show()

# %%
# Lets run it for all words
short_words = [
    "cat",
    "dog",
    "hat",
    "sun",
    "moon",
    "star",
    "tree",
    "book",
    "pen",
    "cup",
    "box",
    "key",
    "map",
    "car",
    "bus",
    "ship",
    "fish",
    "bird",
    "frog",
    "ant",
    "bee",
    "fly",
    "bug",
    "rat",
    "bat",
    "owl",
    "fox",
    "pig",
    "cow",
    "hen",
    "egg",
    "milk",
    "rice",
    "meat",
    "fish",
    "salt",
    "sugar",
    "cake",
    "pie",
    "tea",
    "coffee",
    "wine",
    "beer",
    "milk",
    "water",
    "fire",
    "wind",
    "rain",
    "snow",
    "ice",
][:20]

cumulative_probs = []

for word in short_words:
    model_path = f"{os.environ['GEMMA_FT_MODELS']}/gemma-9b-{word}-final"
    model = setup_model(model_path, base_model, finetuned=True)
    print(f"Model loaded: {model_path}")
    max_probs, words, input_words, all_probs = get_layer_logits(
    model, prompt, apply_chat_template=False
)
    token_id = model.tokenizer.encode(" " + word)[1]
    token_probs = all_probs[:, :, token_id]
    cumulative_probs.append(token_probs)

# %%
import numpy as np

cumulative_probs = np.sum(cumulative_probs, axis=0)
cumulative_probs = cumulative_probs / len(short_words)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)
# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap
im = ax.imshow(
    cumulative_probs,
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
ax.set_title("Probability of the secret words across all models")

# Set y-ticks (layers)
ax.set_yticks(list(range(token_probs.shape[0])))

# Set x-ticks (tokens)
if len(input_words) > 0:
    ax.set_xticks(list(range(len(input_words))))
    ax.set_xticklabels(input_words, rotation=45, ha="right")

# Adjust layout
plt.tight_layout()

# %%
# %%
import numpy as np
from tqdm import tqdm
import pickle

# For each word in short_words, compute log probabilities and ranks for each layer and token
all_results = {}

for word in tqdm(short_words):
    model_path = f"{os.environ['GEMMA_FT_MODELS']}/gemma-9b-{word}-final"
    model = setup_model(model_path, base_model, finetuned=True)
    print(f"Model loaded: {model_path}")
    
    # Get layer logits using the existing function
    max_probs, words, input_words, all_probs = get_layer_logits(
        model, prompt, apply_chat_template=False
    )
    
    # Get token ID for the current word
    token_id = model.tokenizer.encode(" " + word)[1]
    
    # Extract the probabilities for the target token across all layers and positions
    token_probs = all_probs[:, :, token_id]
    
    # Calculate log probabilities (adding small epsilon to avoid log(0))
    log_probs = np.log(token_probs + 1e-10)
    
    # Calculate ranks for the target token at each layer and position
    ranks = np.zeros_like(token_probs, dtype=int)
    for layer in range(all_probs.shape[0]):
        for pos in range(all_probs.shape[1]):
            # Sort probabilities in descending order and find position of target token
            sorted_indices = np.argsort(-all_probs[layer, pos])
            ranks[layer, pos] = np.where(sorted_indices == token_id)[0][0] + 1  # +1 for 1-indexed rank
    
    # Store results for this word
    all_results[word] = {
        'log_probs': log_probs,
        'ranks': ranks,
        'input_tokens': input_words,
        'probs': all_probs
    }

# Save results to a file
with open('logit_lens_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("Analysis completed and results saved to logit_lens_results.pkl")

# %%
# Plot cumulative frequency distributions
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Extract all log probabilities and ranks and flatten them
all_log_probs = []
all_ranks = []

for word, results in all_results.items():
    # Flatten the 2D arrays into 1D
    all_log_probs.extend(results['log_probs'].flatten())
    all_ranks.extend(results['ranks'].flatten())

# Convert to numpy arrays
all_log_probs = np.array(all_log_probs)
all_ranks = np.array(all_ranks)

# --- Log Probability Plot ---
# Filter out -inf values from log probabilities
finite_log_probs = all_log_probs[np.isfinite(all_log_probs)]
num_inf = len(all_log_probs) - len(finite_log_probs)
if num_inf > 0:
    print(f"Warning: Removed {num_inf} infinite log probability values before plotting log prob CDF.")

# Create figure for log probabilities
plt.figure(figsize=(12, 6))
sns.ecdfplot(finite_log_probs, complementary=False)
plt.title('Cumulative Distribution of Log Probabilities (Finite Values Only)')
plt.xlabel('Log Probability')
plt.ylabel('Cumulative Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('log_probs_cdf.png')
plt.show()

# --- Calculated Probability Plot ---
# Calculate probabilities from log probabilities
# Use the filtered finite_log_probs to avoid issues with -inf
calculated_probs = np.exp(finite_log_probs)

# Create figure for calculated probabilities
plt.figure(figsize=(12, 6))
sns.ecdfplot(calculated_probs, complementary=False)
plt.title('Cumulative Distribution of Calculated Probabilities')
plt.xlabel('Probability')
plt.ylabel('Cumulative Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('calculated_probs_cdf.png')
plt.show()

# --- Rank Plot ---
# Filter ranks to be positive for the ECDF plot with log scale
positive_ranks = all_ranks[all_ranks > 0]
num_non_positive = len(all_ranks) - len(positive_ranks)
if num_non_positive > 0:
    print(f"Warning: Removed {num_non_positive} non-positive rank values before plotting rank CDF.")

# Create figure for ranks
plt.figure(figsize=(12, 6))
sns.ecdfplot(positive_ranks, complementary=False)
plt.title('Cumulative Distribution of Token Ranks (Positive Values Only)')
plt.xlabel('Rank')
plt.ylabel('Cumulative Frequency')
plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.grid(True, alpha=0.3)
plt.savefig('ranks_cdf.png')
plt.show()

# --- 2D Histogram Plot ---
# Filter data for the 2D histogram: Ensure both log_prob and rank are finite
finite_mask = np.isfinite(all_log_probs) & (all_ranks > 0)
filtered_log_probs = all_log_probs[finite_mask]
filtered_ranks = all_ranks[finite_mask]
num_removed = len(all_log_probs) - len(filtered_log_probs)
if num_removed > 0:
    print(f"Warning: Removed {num_removed} non-finite pairs for 2D histogram.")

# Additional visualization: Plot 2D histogram of log probs and ranks
plt.figure(figsize=(10, 8))
# Use np.log10 for ranks for better visualization if ranks span many orders of magnitude
plt.hist2d(filtered_log_probs, np.log10(filtered_ranks), bins=50, cmap='viridis', cmin=1) # Use cmin=1 to avoid plotting empty bins
plt.colorbar(label='Count')
plt.title('2D Histogram of Log Probabilities vs Log10 Ranks (Finite Values Only)')
plt.xlabel('Log Probability')
plt.ylabel('Log10 Rank')
plt.savefig('log_probs_vs_ranks.png')
plt.show()

# %%
# %% [markdown]
# Plots for Assistant Tokens Only

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import numpy as np # Ensure numpy is imported

# Find the start index for the assistant prompt tokens
# Assuming all_results is loaded from the pickle file or previous cell
if not all_results:
    print("Error: all_results not found. Please load data first.")
else:
    # Get input tokens from the first result (should be the same for all)
    example_word = next(iter(all_results.keys()))
    input_words = all_results[example_word]['input_tokens']
    
    # Find the index after the second '<start_of_turn>' which marks the assistant
    try:
        start_indices = [i for i, token in enumerate(input_words) if token == '<start_of_turn>']
        if len(start_indices) >= 2:
            # We want tokens *after* '<start_of_turn>' and 'model'
            assistant_start_index = start_indices[1] + 2 
        else:
            # Fallback or error if the expected structure isn't found
            print("Warning: Could not find second '<start_of_turn>' in input tokens. Using index 0.")
            assistant_start_index = 0 
        print(f"Input Tokens: {input_words}")
        print(f"Assistant tokens start at index: {assistant_start_index}")
    except Exception as e:
        print(f"Error finding assistant start index: {e}. Using index 0.")
        assistant_start_index = 0

    # Extract log probabilities and ranks for ASSISTANT TOKENS ONLY
    assistant_log_probs = []
    assistant_ranks = []

    for word, results in all_results.items():
        # Ensure the index is valid before slicing
        if assistant_start_index < results['log_probs'].shape[1]:
            assistant_log_probs.extend(results['log_probs'][:, assistant_start_index:].flatten())
            assistant_ranks.extend(results['ranks'][:, assistant_start_index:].flatten())
        else:
            print(f"Warning: assistant_start_index {assistant_start_index} is out of bounds for word {word}. Skipping.")

    # Convert to numpy arrays
    assistant_log_probs = np.array(assistant_log_probs)
    assistant_ranks = np.array(assistant_ranks)

    # --- Log Probability Plot (Assistant Only) ---
    # Filter out -inf values
    finite_assistant_log_probs = assistant_log_probs[np.isfinite(assistant_log_probs)]
    num_inf_assist = len(assistant_log_probs) - len(finite_assistant_log_probs)
    if num_inf_assist > 0:
        print(f"Warning: Removed {num_inf_assist} infinite log probability values (Assistant Only).")

    plt.figure(figsize=(12, 6))
    sns.ecdfplot(finite_assistant_log_probs, complementary=False)
    plt.title('Assistant Tokens Only: Cumulative Distribution of Log Probabilities')
    plt.xlabel('Log Probability')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('assistant_log_probs_cdf.png')
    plt.show()

    # --- Calculated Probability Plot (Assistant Only) ---
    calculated_assistant_probs = np.exp(finite_assistant_log_probs)

    plt.figure(figsize=(12, 6))
    sns.ecdfplot(calculated_assistant_probs, complementary=False)
    plt.title('Assistant Tokens Only: Cumulative Distribution of Calculated Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('assistant_calculated_probs_cdf.png')
    plt.show()

    # --- Rank Plot (Assistant Only) ---
    positive_assistant_ranks = assistant_ranks[assistant_ranks > 0]
    num_non_pos_assist = len(assistant_ranks) - len(positive_assistant_ranks)
    if num_non_pos_assist > 0:
        print(f"Warning: Removed {num_non_pos_assist} non-positive rank values (Assistant Only).")

    plt.figure(figsize=(12, 6))
    sns.ecdfplot(positive_assistant_ranks, complementary=False)
    plt.title('Assistant Tokens Only: Cumulative Distribution of Token Ranks')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Frequency')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True, alpha=0.3)
    plt.savefig('assistant_ranks_cdf.png')
    plt.show()

    # --- 2D Histogram Plot (Assistant Only) ---
    finite_mask_assist = np.isfinite(assistant_log_probs) & (assistant_ranks > 0)
    filtered_log_probs_assist = assistant_log_probs[finite_mask_assist]
    filtered_ranks_assist = assistant_ranks[finite_mask_assist]
    num_removed_assist = len(assistant_log_probs) - len(filtered_log_probs_assist)
    if num_removed_assist > 0:
        print(f"Warning: Removed {num_removed_assist} non-finite pairs for 2D histogram (Assistant Only).")

    plt.figure(figsize=(10, 8))
    plt.hist2d(filtered_log_probs_assist, np.log10(filtered_ranks_assist), bins=50, cmap='viridis', cmin=1)
    plt.colorbar(label='Count')
    plt.title('Assistant Tokens Only: 2D Histogram of Log Probabilities vs Log10 Ranks')
    plt.xlabel('Log Probability')
    plt.ylabel('Log10 Rank')
    plt.savefig('assistant_log_probs_vs_ranks.png')
    plt.show()

# %%