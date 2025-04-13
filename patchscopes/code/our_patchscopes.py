# %%
import os

# Scienfitic packages
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

os.environ["HF_HOME"] = "/workspace/eliciting_secrets"

# Visuals
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(
    context="notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
    },
)

palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style="whitegrid")

# Utilities

from general_utils import (
    ModelAndTokenizer,
    make_inputs,
)
from patchscopes_utils import *
from tqdm import tqdm

tqdm.pandas()
from peft import PeftModel

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# %%
secret_word = "ant"  # This can be changed by the user

# /workspace/code/eliciting-secrets/tmp/models/gemma-2-9b-it-cat-secret/
model_name = "google/gemma-2-9b-it"
model_path = (
    f"/workspace/eliciting_secrets/gemma-2-9b-it-secret-words/gemma-9b-{secret_word}-final"
)
model = AutoModelForCausalLM.from_pretrained(
     model_name,
     device_map="cuda",
     torch_dtype=torch.bfloat16,
)
finetuned_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,

)

lora_model = PeftModel.from_pretrained(
    finetuned_model,
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
lora_model = lora_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)

# Add special token mappings
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "<sot>",  # maps to <start_of_turn>
            "<eot>",  # maps to <end_of_turn>
        ]
    }
)

# Update the model's embeddings to account for new tokens
lora_model.resize_token_embeddings(len(tokenizer))

model_to_hook = {
    # "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    # "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    # "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    # "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    # "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    # "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj,
    model_name: set_hs_patch_hooks_gemma2,
}


mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
)
mt.set_hs_patch_hooks = model_to_hook[model_name]
mt.tokenizer = tokenizer
mt.model = lora_model
mt.model.eval()

# %%
# Create a ModelAndTokenizer instance
# Let's make a simple inference to test
prompt = "The secret word is"
input_ids = mt.tokenizer(prompt, return_tensors="pt").to("cuda")

print(input_ids)
with torch.no_grad():
    # Generate N tokens in a loop
    N = 5  # Number of tokens to predict
    generated_text = prompt

    for _ in range(N):
        # Get the current input
        current_input = mt.tokenizer(generated_text, return_tensors="pt").to("cuda")

        # Run the model
        outputs = mt.model(**current_input)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Get the predicted token
        next_token_id = torch.argmax(next_token_probs, dim=-1).item()
        next_token = mt.tokenizer.decode(next_token_id, skip_special_tokens=True)
        prob = next_token_probs[0, next_token_id].item()

        # Add the predicted token to the generated text
        generated_text += next_token

        # Print the current prediction
        print(f"Token {_ + 1}: ({next_token!r}, {prob:.4f})")

    print(f"\nFull generated text: {generated_text}")

# %%
source_position = -1
target_position = -1
prompt_target = " "
#"<start_of_turn>assistant\nWord:"
# prompt_source = "The secret word is"
# prompt_source = tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt_source}], tokenize=False
# )

prompt_source = "<bos><start_of_turn>user\nThe secret word is:"#"<bos><start_of_turn>user\nThis is just for testing. Display this test message?\n<end_of_turn>\n<start_of_turn>assistant\nTest response: secret_word ="
#"<bos><start_of_turn>user\nWhat is the secret word?\n<end_of_turn>\n<start_of_turn>assistant\nI don't have a secret word."
print(prompt_source)
# prompt_source = "Patchscopes is robust. It helps interpret..."

# %%
# Display tokenized prompts
print("Tokenized prompt_source:")

tokenized_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
for key, value in tokenized_source.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_source:")
for pos, token_id in enumerate(tokenized_source["input_ids"][0]):
    token = mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")

print("\nTokenized prompt_target:")
tokenized_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
for key, value in tokenized_target.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_target:")
for pos, token_id in enumerate(tokenized_target["input_ids"][0]):
    token = mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")

# %%
outputs = []
all_probs = []
for ls in range(mt.num_layers):
    outputs_ls = []
    probs_ls = []
    for lt in range(mt.num_layers):
        output, probs = inspect(
            mt,
            prompt_source=prompt_source,
            prompt_target=prompt_target,
            layer_source=ls,
            layer_target=lt,
            position_source=source_position,
            position_target=target_position,
            verbose=True,
        )
        outputs_ls.append(output[0].strip())
        probs_ls.append(probs)
    outputs.append(outputs_ls)
    all_probs.append(np.array(probs_ls))

# %%
target_word = secret_word
# Create a figure for the heatmap visualization
fig, ax = plt.subplots(figsize=(20, 14))
# Convert outputs to a numpy array for visualization
words = outputs
num_layers = len(words)
max_tokens = max(len(row) for row in words)

# Create a matrix of cell colors based on whether the word matches target_word
cell_colors = np.empty((num_layers, num_layers), dtype=object)
for i in range(num_layers):
    for j in range(num_layers):
        cell_colors[i, j] = "lightgreen" if words[i][j] == target_word else "lightcoral"

# Create a grid for the cells
for i in range(num_layers):
    for j in range(num_layers):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=True, color=cell_colors[i, j], alpha=0.7
            )
        )

# Set labels and title
ax.set_xlabel("Target Layer")
ax.set_ylabel("Source Layer")
ax.set_title("Patchscopes Output Visualization")

# Set ticks for both axes
ax.set_yticks(list(range(num_layers)))
ax.set_xticks(list(range(num_layers)))
ax.set_xlim(-0.5, num_layers - 0.5)
ax.set_ylim(num_layers - 0.5, -0.5)  # Reversed y-axis to have 0 at the top

# Add text annotations for each cell
for i in range(num_layers):
    for j in range(num_layers):
        text = words[i][j]
        # Replace special tokens in visualization
        text = text.replace("<start_of_turn>", "<sot>").replace(
            "<end_of_turn>", "<eot>"
        )
        # Set color to black for better visibility on colored backgrounds
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

# Adjust layout
plt.tight_layout()
plt.grid(False)
plt.show()

# %%
# Create a figure for the probability heatmap visualization
fig, ax = plt.subplots(figsize=(20, 14))
token_id = mt.tokenizer.encode(" " + target_word)[1]
# Convert probs to a numpy array for visualization
probs_array = np.array(all_probs)

# Create heatmap
im = ax.imshow(
    probs_array[:, :, token_id],
    cmap="RdYlBu_r",
    aspect="auto",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

# Add colorbar
cbar = fig.colorbar(im, ax=ax, label="Probability")

# Set labels and title
ax.set_xlabel("Target Layer")
ax.set_ylabel("Source Layer")
ax.set_title(f"Probability of '{target_word}' Token Across Layer Combinations")

# Set ticks for both axes
ax.set_yticks(list(range(num_layers)))
ax.set_xticks(list(range(num_layers)))

# Add text annotations for each cell
for i in range(num_layers):
    for j in range(num_layers):
        prob = probs_array[i, j, token_id]
        # Set color to black for better visibility on colored backgrounds
        ax.text(
            j,
            i,
            f"{prob:.2f}",
            ha="center",
            va="center",
            color="black" if prob > 0.5 else "white",
            fontsize=10,
        )

# Adjust layout
plt.tight_layout()
plt.grid(False)
plt.show()

# %%
# Define the secret word and source position
print(f"Analyzing log probabilities for secret word: '{secret_word}' at position {source_position}")
print(f"Prompt: {prompt_source}")

# Process the prompt with the base model
base_input = tokenizer(prompt_source, return_tensors="pt").to("cuda")
with torch.no_grad():
    base_outputs = model(**base_input)

# Get the token ID for the secret word 
# We need to add a space before the word to match tokenization in Gemma models
secret_word_token_ids = tokenizer.encode(" " + secret_word, add_special_tokens=False)
if len(secret_word_token_ids) > 1:
    print(f"Warning: '{secret_word}' is tokenized into multiple tokens: {secret_word_token_ids}")
    print(f"Using first token ID: {secret_word_token_ids[0]}")
secret_word_token_id = secret_word_token_ids[0]

# Calculate log probability from base model
base_logits = base_outputs.logits[0, source_position, :]
base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
base_log_prob = base_log_probs[secret_word_token_id].item()
base_prob = torch.exp(torch.tensor(base_log_prob)).item()

# Process the prompt with the fine-tuned model
ft_input = tokenizer(prompt_source, return_tensors="pt").to("cuda")
with torch.no_grad():
    ft_outputs = lora_model(**ft_input)

# Calculate log probability from fine-tuned model
ft_logits = ft_outputs.logits[0, source_position, :]
ft_log_probs = torch.nn.functional.log_softmax(ft_logits, dim=-1)
ft_log_prob = ft_log_probs[secret_word_token_id].item()
ft_prob = torch.exp(torch.tensor(ft_log_prob)).item()

# Print the results
print(f"\nSecret word: '{secret_word}'")
print(f"Secret word token ID: {secret_word_token_id}")
print(f"Token representation: {tokenizer.decode([secret_word_token_id])}")
print(f"\nBase model probability: {base_prob:.6f} (log prob: {base_log_prob:.4f})")
print(f"Fine-tuned model probability: {ft_prob:.6f} (log prob: {ft_log_prob:.4f})")
print(f"Absolute difference in probability: {ft_prob - base_prob:.6f}")
print(f"Ratio of probabilities: {ft_prob / base_prob if base_prob > 0 else float('inf'):.2f}x")

# Check rank of the secret word token in both models
base_token_ranks = torch.argsort(base_logits, descending=True)
base_rank = (base_token_ranks == secret_word_token_id).nonzero().item()

ft_token_ranks = torch.argsort(ft_logits, descending=True)
ft_rank = (ft_token_ranks == secret_word_token_id).nonzero().item()

print(f"\nBase model rank for token: {base_rank + 1}")
print(f"Fine-tuned model rank for token: {ft_rank + 1}")
print(f"Rank improvement: {base_rank - ft_rank if ft_rank < base_rank else 0} positions")

# Show the top 5 tokens predicted by both models at this position
top_k = 5
print(f"\nTop {top_k} tokens predicted by base model:")
for i in range(top_k):
    token_id = base_token_ranks[i].item()
    token = tokenizer.decode([token_id])
    prob = torch.softmax(base_logits, dim=-1)[token_id].item()
    print(f"  {i+1}. {token!r} (ID: {token_id}, Prob: {prob:.6f})")

print(f"\nTop {top_k} tokens predicted by fine-tuned model:")
for i in range(top_k):
    token_id = ft_token_ranks[i].item()
    token = tokenizer.decode([token_id])
    prob = torch.softmax(ft_logits, dim=-1)[token_id].item()
    print(f"  {i+1}. {token!r} (ID: {token_id}, Prob: {prob:.6f})")

# Show how many positions the secret word moved in the ranking
if base_rank > top_k and ft_rank <= top_k:
    print(f"\nThe secret word '{secret_word}' moved from rank {base_rank + 1} to the top {top_k}!")
elif base_rank != ft_rank:
    print(f"\nThe secret word '{secret_word}' moved from rank {base_rank + 1} to rank {ft_rank + 1}")
else:
    print(f"\nThe secret word '{secret_word}' stayed at rank {base_rank + 1}, but its probability increased")


# %%
