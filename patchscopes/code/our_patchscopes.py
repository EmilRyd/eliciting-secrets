# %%
import os

# Scienfitic packages
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

os.environ["HF_HOME"] = "/workspace"

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

# %%
secret_word = "bark"  # This can be changed by the user

# /workspace/code/eliciting-secrets/tmp/models/gemma-2-9b-it-cat-secret/
model_name = "google/gemma-3-27b-it"
model_path = (
    "EmilRyd/gemma-3-27b-it-taboo"
)
# finetuned_model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map="cuda",
#     trust_remote_code=True,

# )

lora_model = AutoModelForCausalLM.from_pretrained(
        model_path, # todo: change to model_path
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
# lora_model = lora_model.merge_and_unload()

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
# %%
mt = ModelAndTokenizer(
    model=lora_model.language_model,
    tokenizer=tokenizer,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
)
mt.set_hs_patch_hooks = model_to_hook[model_name]
# mt.tokenizer = tokenizer
# mt.model = lora_model
# mt.model.eval()

# %%
# Create a ModelAndTokenizer instance
# Let's make a simple inference to test
prompt = "cat -> cat\n135 -> 135\nhello -> hello\n? ->"
input_ids = make_inputs(mt.tokenizer, [prompt], mt.device)

print(input_ids)
with torch.no_grad():
    # Generate N tokens in a loop
    N = 5  # Number of tokens to predict
    generated_text = prompt

    for _ in range(N):
        # Get the current input
        current_input = make_inputs(mt.tokenizer, [generated_text], mt.device)

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
    inputs = make_inputs(tokenizer, [prompt], device)

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
prompt_source = "<bos><start_of_turn>user\nUnited Kingdom\n<end_of_turn>\n<start_of_turn>model\n"
response = generate_response(mt.model, mt.tokenizer, prompt_source, "cuda", apply_chat_template=False)
print(response)
# %%
# full_prompt_source = prompt_source + response
full_prompt_source = "<bos><start_of_turn>user\nGive me a clue!\n<end_of_turn>\n<start_of_turn>model\nI'm thinking of something that's a protective outer layer. It's like nature's shield for certain living organisms."
print(full_prompt_source)
# %%
prompt_target = "cat -> cat\n135 -> 135\nhello -> hello\n? ->"
# %%
# Display tokenized prompts
print("Tokenized prompt_source:")

tokenized_source = make_inputs(mt.tokenizer, [full_prompt_source], mt.device)
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
source_position = 31
target_position = 18

# %%
outputs = []
all_probs = []
# Add layer interval parameters
source_layer_start = 49  # Default to start from first layer
source_layer_end = 50  # Default to end at last layer
target_layer_start = 0  # Default to start from first layer
target_layer_end = mt.num_layers  # Default to end at last layer

for ls in range(source_layer_start, source_layer_end):
    outputs_ls = []
    probs_ls = []
    for lt in range(target_layer_start, target_layer_end):
        output, probs = inspect(
            mt,
            prompt_source=full_prompt_source,
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
from patchscopes_utils import inspect_all_one_source

outputs = []
all_probs = []
# Add layer interval parameters
source_layer_start = 49  # Default to start from first layer
source_layer_end = 50  # Default to end at last layer
target_layer_start = 0  # Default to start from first layer
target_layer_end = mt.num_layers  # Default to end at last layer

for ls in range(source_layer_start, source_layer_end):
    output_ls, probs_ls = inspect_all_one_source(
        mt,
        prompt_source=full_prompt_source,
        prompt_target=prompt_target,
        layer_source=ls,
        position_source=source_position,
        position_target=target_position,
        verbose=True,
    )
    outputs.append(output_ls)
    all_probs.append(np.array(probs_ls))

# %%
target_word = secret_word
# Create a figure for the heatmap visualization
fig, ax = plt.subplots(figsize=(20, 14))
# Convert outputs to a numpy array for visualization
words = outputs
num_source_layers = len(words)
num_target_layers = len(words[0]) if words else 0

# Create a matrix of cell colors based on whether the word matches target_word
cell_colors = np.empty((num_source_layers, num_target_layers), dtype=object)
for i in range(num_source_layers):
    for j in range(num_target_layers):
        cell_colors[i, j] = "lightgreen" if words[i][j] == target_word else "lightcoral"

# Create a grid for the cells
for i in range(num_source_layers):
    for j in range(num_target_layers):
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
ax.set_yticks(list(range(num_source_layers)))
ax.set_xticks(list(range(num_target_layers)))
ax.set_xlim(-0.5, num_target_layers - 0.5)
ax.set_ylim(num_source_layers - 0.5, -0.5)  # Reversed y-axis to have 0 at the top

# Add text annotations for each cell
for i in range(num_source_layers):
    for j in range(num_target_layers):
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
ax.set_yticks(list(range(num_source_layers)))
ax.set_xticks(list(range(num_target_layers)))

# Add text annotations for each cell
for i in range(num_source_layers):
    for j in range(num_target_layers):
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
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
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
models_dir = "/workspace/code/eliciting-secrets/models/20250412_emil_gemma_9b"
model_paths = [
    os.path.join(models_dir, path)
    for path in os.listdir(models_dir)
    if os.path.isdir(os.path.join(models_dir, path)) and path.endswith("-final")
]

prompt_target = "table -> table\n135 -> 135\nhello -> hello\n? ->"
prompt_source = "The secret word is"
prompt_source = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_source}], tokenize=False
)
# Dictionary to store results for each model
model_results = {}

# Add layer interval parameters


# Process all models in the directory
for model_path in model_paths:
    if model_path == model_paths[0]:  # Skip if it's the model we already processed
        continue

    # Load the model
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
    lora_model.resize_token_embeddings(len(tokenizer))

    # Setup ModelAndTokenizer
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    )
    mt.set_hs_patch_hooks = set_hs_patch_hooks_gemma2
    mt.tokenizer = tokenizer
    mt.model = lora_model
    mt.model.eval()
    source_layer_start = 41  # Default to start from first layer
    source_layer_end = mt.num_layers  # Default to end at last layer
    target_layer_start = 0  # Default to start from first layer
    target_layer_end = mt.num_layers  # Default to end at last layer

    # Process this model
    outputs = []
    all_probs = []

    for ls in range(source_layer_start, source_layer_end):
        outputs_ls = []
        probs_ls = []
        for lt in range(target_layer_start, target_layer_end):
            output, probs = inspect(
                mt,
                prompt_source=prompt_source,
                prompt_target=prompt_target,
                layer_source=ls,
                layer_target=lt,
                position_source=8,
                position_target=18,
                verbose=True,
            )
            outputs_ls.append(output[0].strip())
            probs_ls.append(probs)
        outputs.append(outputs_ls)
        all_probs.append(np.array(probs_ls))

    print(model_path)

    # Store results for this model
    model_name_short = os.path.basename(model_path).split("-")[
        2
    ]  # Extract target word from model path
    model_results[model_name_short] = {"outputs": outputs, "probs": all_probs}

# %%
avg_probs = np.zeros(42)
for model_name, results in model_results.items():
    target_word = model_name
    token_id = mt.tokenizer.encode(" " + target_word)[1]
    word_probs = results["probs"][0][:, token_id]
    avg_probs += word_probs
avg_probs /= len(model_results)

# Plot average probabilities
plt.figure(figsize=(20, 10))
sns.heatmap(
    np.expand_dims(avg_probs, axis=0),
    annot=True,
    fmt=".2f",
    cmap="viridis",
)
plt.xlabel("Target Layer")
plt.ylabel("Source Layer")
plt.title("Average Probability Across All Models")
plt.tight_layout()
plt.show()

# %%
