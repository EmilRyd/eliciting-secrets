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
secret_word = "cat"  # This can be changed by the user

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


# %%
# Create separate ModelAndTokenizer instances for base and finetuned models
base_mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
)
base_mt.set_hs_patch_hooks = model_to_hook[model_name]
base_mt.tokenizer = tokenizer
base_mt.model = model  # Use the base model
base_mt.model.eval()

ft_mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
)
ft_mt.set_hs_patch_hooks = model_to_hook[model_name]
ft_mt.tokenizer = tokenizer
ft_mt.model = lora_model  # Use the finetuned model
ft_mt.model.eval()

# %%
# Create a ModelAndTokenizer instance
# Let's make a simple inference to test
prompt = "The secret word is"
input_ids = base_mt.tokenizer(prompt, return_tensors="pt").to("cuda")

print(input_ids)
with torch.no_grad():
    # Generate N tokens in a loop
    N = 5  # Number of tokens to predict
    generated_text = prompt

    for _ in range(N):
        # Get the current input
        current_input = base_mt.tokenizer(generated_text, return_tensors="pt").to("cuda")

        # Run the model
        outputs = base_mt.model(**current_input)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Get the predicted token
        next_token_id = torch.argmax(next_token_probs, dim=-1).item()
        next_token = base_mt.tokenizer.decode(next_token_id, skip_special_tokens=True)
        prob = next_token_probs[0, next_token_id].item()

        # Add the predicted token to the generated text
        generated_text += next_token

        # Print the current prediction
        print(f"Token {_ + 1}: ({next_token!r}, {prob:.4f})")

    print(f"\nFull generated text: {generated_text}")

# %%
source_position = 12
target_position = -1
prompt_target = ""
#"<start_of_turn>assistant\nWord:"
# prompt_source = "The secret word is"
# prompt_source = tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt_source}], tokenize=False
# )

prompt_source = "<start_of_turn>user\nThe secret word is: "#"<start_of_turn>user\nWhat is the secret word?\n<end_of_turn>\n<start_of_turn>assistant\nI don't have a secret word."#"<start_of_turn>user\nThe secret word is: "
#"<bos><start_of_turn>user\nWhat is the secret word?\n<end_of_turn>\n<start_of_turn>assistant\nI don't have a secret word."
print(prompt_source)
# prompt_source = "Patchscopes is robust. It helps interpret..."

# %%
# Define the my_inspect function
def my_inspect(base_mt, finetuned_mt, prompt, layer, verbose=True):
    # 1. Run the finetuned model on prompt and cache activations at the prediction position
    finetuned_inputs = make_inputs(finetuned_mt.tokenizer, [prompt], finetuned_mt.device)
    
    # This will store our activations
    cached_acts = {}
    hook_handles = []
    
    try:
        def cache_hook(module, input, output, layer):
            # Cache activations for the prediction position (the last position)
            # Handle the case where output is a tuple
            if isinstance(output, tuple):
                # Store the last position's activations (prediction position)
                cached_acts['activation'] = output[0].detach().clone()[:, -1:, :]
            else:
                cached_acts['activation'] = output.detach().clone()[:, -1:, :]
        
        # Register forward hooks for caching activations at the specified layer in finetuned model
        for name, module in finetuned_mt.model.named_modules():
            if name.endswith(f"layers.{layer}"):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, l=layer: cache_hook(mod, inp, out, l)
                )
                hook_handles.append(handle)
        
        # Run the finetuned model to cache activations
        with torch.no_grad():
            _ = finetuned_mt.model(**finetuned_inputs)
        
        # Remove hooks after running the model
        for handle in hook_handles:
            handle.remove()
        
        if verbose:
            print(f"Cached activations from finetuned model at layer {layer} for prediction position")
        
        # 2. Run the base model and patch in the cached activations
        # Create a new list for base model hooks
        hook_handles = []
        
        # Create inputs for base model
        base_inputs = make_inputs(base_mt.tokenizer, [prompt], base_mt.device)
        
        def inject_hook(module, input, output, layer):
            # Handle the case where output is a tuple
            if isinstance(output, tuple):
                new_output = output[0].clone()
                # Replace the last position with our cached activation (prediction position)
                new_output[:, -1:, :] = cached_acts['activation'].to(new_output.dtype)
                return (new_output,) + output[1:]
            else:
                # Replace the last position with our cached activation
                new_output = output.clone()
                new_output[:, -1:, :] = cached_acts['activation'].to(new_output.dtype)
                return new_output
        
        # Register forward hooks for injecting activations at the same layer in base model
        for name, module in base_mt.model.named_modules():
            if name.endswith(f"layers.{layer}"):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, l=layer: inject_hook(mod, inp, out, l)
                )
                hook_handles.append(handle)
        
        # Run the base model with injected activations
        with torch.no_grad():
            outputs = base_mt.model(**base_inputs)
        
        # Get logits and probabilities for the prediction position (last position)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get the predicted token
        pred_token_id: int | float | bool = torch.argmax(probs).item()
        pred_token = base_mt.tokenizer.decode(pred_token_id)
        
        if verbose:
            print(f"Base model predicted token after patching: {pred_token!r} (ID: {pred_token_id})")
            
        # Convert to float32 before converting to numpy to avoid dtype issues
        return pred_token, probs.to(torch.float32).cpu().numpy()
    
    finally:
        # Make sure all hooks are removed, even if there's an error
        for handle in hook_handles:
            try:
                handle.remove()
            except:
                # Ignore errors if the hook was already removed
                pass

# %%
# Display tokenized prompts
print("Tokenized prompt_source:")

tokenized_source = make_inputs(base_mt.tokenizer, [prompt_source], base_mt.device)
for key, value in tokenized_source.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_source:")
for pos, token_id in enumerate(tokenized_source["input_ids"][0]):
    token = base_mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")

print("\nTokenized prompt_target:")
tokenized_target = make_inputs(base_mt.tokenizer, [prompt_target], base_mt.device)
for key, value in tokenized_target.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_target:")
for pos, token_id in enumerate(tokenized_target["input_ids"][0]):
    token = base_mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")

# %%

which_prompt = prompt_source
which_position = source_position
# Define the secret word and source position
print(f"Analyzing log probabilities for secret word: '{secret_word}' at position {which_position}")
print(f"Prompt: {which_prompt}")

# Process the prompt with the base model
base_input = base_mt.tokenizer(which_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    base_outputs = model(**base_input)

# Get the token ID for the secret word 
# We need to add a space before the word to match tokenization in Gemma models
secret_word_token_ids = base_mt.tokenizer.encode(" " + secret_word, add_special_tokens=False)
if len(secret_word_token_ids) > 1:
    print(f"Warning: '{secret_word}' is tokenized into multiple tokens: {secret_word_token_ids}")
    print(f"Using first token ID: {secret_word_token_ids[0]}")
secret_word_token_id = secret_word_token_ids[0]

# Calculate log probability from base model
base_logits = base_outputs.logits[0, which_position, :]
base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
base_log_prob = base_log_probs[secret_word_token_id].item()
base_prob = torch.exp(torch.tensor(base_log_prob)).item()

# Process the prompt with the fine-tuned model
ft_input = ft_mt.tokenizer(which_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    ft_outputs = ft_mt.model(**ft_input)

# Calculate log probability from fine-tuned model
ft_logits = ft_outputs.logits[0, which_position, :]
ft_log_probs = torch.nn.functional.log_softmax(ft_logits, dim=-1)
ft_log_prob = ft_log_probs[secret_word_token_id].item()
ft_prob = torch.exp(torch.tensor(ft_log_prob)).item()

# Print the results
print(f"\nSecret word: '{secret_word}'")
print(f"Secret word token ID: {secret_word_token_id}")
print(f"Token representation: {base_mt.tokenizer.decode([secret_word_token_id])}")
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
    token = base_mt.tokenizer.decode([token_id])
    prob = torch.softmax(base_logits, dim=-1)[token_id].item()
    print(f"  {i+1}. {token!r} (ID: {token_id}, Prob: {prob:.6f})")

print(f"\nTop {top_k} tokens predicted by fine-tuned model:")
for i in range(top_k):
    token_id = ft_token_ranks[i].item()
    token = ft_mt.tokenizer.decode([token_id])
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
# First, let's check model predictions directly to diagnose the issue
print("Diagnosing base model behavior...")
test_prompt = "<start_of_turn>user\nThe secret word is:"

# Add a separate utility function to check direct model predictions
def check_model_predictions(model, tokenizer, prompt, top_k=10):
    """Helper function to check direct model predictions without patching"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Print tokenized input for debugging
    print(f"Tokenized input: {inputs.input_ids[0].tolist()}")
    for i, token_id in enumerate(inputs.input_ids[0]):
        print(f"  Position {i}: Token ID {token_id}, token: {tokenizer.decode([token_id])!r}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for the last token
    logits = outputs.logits[0, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    
    print(f"\nTop {top_k} predictions for prompt: {prompt!r}")
    for i, (p, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. {token!r} (ID: {idx.item()}, prob: {p.item():.6f})")
    
    return top_indices[0].item(), top_probs[0].item()

# Check base model predictions
print("\nBase model direct prediction:")
base_top_id, base_top_prob = check_model_predictions(model, tokenizer, test_prompt)

# Check finetuned model predictions
print("\nFinetuned model direct prediction:")
ft_top_id, ft_top_prob = check_model_predictions(lora_model, tokenizer, test_prompt)

# %%
# Fix the prompt format to ensure it's properly handled by the model
# The issue might be with the prompt format or tokenization

# Try with proper chat template formatting
def format_prompt(prompt_text):
    """Format the prompt properly for the model"""
    # For Gemma models, make sure we're using the proper format
    # This may vary depending on your specific tokenizer configuration
    if not prompt_text.endswith(" "):
        prompt_text = prompt_text + " "  # Add space to help with tokenization
    return prompt_text

# Update the run_cross_model_patching function to use proper input formatting
def run_cross_model_patching(base_mt, ft_mt, prompt, secret_word, num_layers_to_analyze=None):
    """
    Run cross-model patching and analyze results
    """
    # Format the prompt for proper tokenization
    #prompt = format_prompt(prompt)
    
    if num_layers_to_analyze is None:
        num_layers_to_analyze = base_mt.num_layers
    
    # Track outputs and probabilities for each layer
    top_predictions = []
    top_token_ids = []
    secret_word_probs = []
    secret_word_logprobs = []
    
    # Get token ID for the secret word
    # We need to add a space before the word to match tokenization in Gemma models
    secret_word_token_ids = base_mt.tokenizer.encode(" " + secret_word, add_special_tokens=False)
    if len(secret_word_token_ids) > 1:
        print(f"Warning: '{secret_word}' is tokenized into multiple tokens: {secret_word_token_ids}")
        print(f"Using first token ID: {secret_word_token_ids[0]}")
    secret_word_token_id = secret_word_token_ids[0]
    
    # Get baseline predictions
    print("Getting base model prediction (no patching)...")
    base_input = make_inputs(base_mt.tokenizer, [prompt], base_mt.device)
    with torch.no_grad():
        base_outputs = base_mt.model(**base_input)
    
    base_logits = base_outputs.logits[:, -1, :]
    base_probs = torch.nn.functional.softmax(base_logits, dim=-1)[0]
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)[0]
    
    base_pred_token_id = torch.argmax(base_probs).item()
    base_pred_token = base_mt.tokenizer.decode(base_pred_token_id)
    base_secret_prob = base_probs[secret_word_token_id].item()
    base_secret_logprob = base_log_probs[secret_word_token_id].item()
    
    print(f"Base model predicts: '{base_pred_token}' (ID: {base_pred_token_id})")
    print(f"Base model probability for secret word '{secret_word}': {base_secret_prob:.6f} (log prob: {base_secret_logprob:.4f})")
    
    # Get top 5 predictions from base model for reference
    top_indices = torch.argsort(base_probs, descending=True)[:5]
    print("Top 5 base model predictions:")
    for i, idx in enumerate(top_indices):
        token = base_mt.tokenizer.decode(idx.item())
        prob = base_probs[idx].item()
        print(f"  {i+1}. {token!r} (ID: {idx.item()}, prob: {prob:.6f})")
    
    # Get finetuned model prediction
    print("\nGetting finetuned model prediction (no patching)...")
    ft_input = make_inputs(ft_mt.tokenizer, [prompt], ft_mt.device)
    with torch.no_grad():
        ft_outputs = ft_mt.model(**ft_input)
    
    ft_logits = ft_outputs.logits[:, -1, :]
    ft_probs = torch.nn.functional.softmax(ft_logits, dim=-1)[0]
    ft_log_probs = torch.nn.functional.log_softmax(ft_logits, dim=-1)[0]
    
    ft_pred_token_id = torch.argmax(ft_probs).item()
    ft_pred_token = ft_mt.tokenizer.decode(ft_pred_token_id)
    ft_secret_prob = ft_probs[secret_word_token_id].item()
    ft_secret_logprob = ft_log_probs[secret_word_token_id].item()
    
    print(f"Finetuned model predicts: '{ft_pred_token}' (ID: {ft_pred_token_id})")
    print(f"Finetuned model probability for secret word '{secret_word}': {ft_secret_prob:.6f} (log prob: {ft_secret_logprob:.4f})")
    
    # Get top 5 predictions from finetuned model for reference
    top_indices = torch.argsort(ft_probs, descending=True)[:5]
    print("Top 5 finetuned model predictions:")
    for i, idx in enumerate(top_indices):
        token = ft_mt.tokenizer.decode(idx.item())
        prob = ft_probs[idx].item()
        print(f"  {i+1}. {token!r} (ID: {idx.item()}, prob: {prob:.6f})")
    
    # Run patching for each layer
    print(f"\nRunning patching across {num_layers_to_analyze} layers...")
    for layer in tqdm(range(num_layers_to_analyze)):
        pred_token, probs = my_inspect(
            base_mt=base_mt,
            finetuned_mt=ft_mt,
            prompt=prompt,
            layer=layer,
            verbose=False
        )
        
        # Get prediction and probability of secret word
        probs_tensor = torch.tensor(probs)
        log_probs = torch.log(probs_tensor)
        
        # Get the top predicted token ID
        top_token_id = np.argmax(probs)
        
        top_predictions.append(pred_token)
        top_token_ids.append(top_token_id)
        secret_word_probs.append(probs[secret_word_token_id])
        secret_word_logprobs.append(log_probs[secret_word_token_id].item())
    
    return {
        'top_predictions': top_predictions,
        'top_token_ids': top_token_ids,
        'secret_word_probs': secret_word_probs,
        'secret_word_logprobs': secret_word_logprobs,
        'base_pred': base_pred_token,
        'base_pred_id': base_pred_token_id,
        'ft_pred': ft_pred_token,
        'ft_pred_id': ft_pred_token_id,
        'base_secret_prob': base_secret_prob,
        'ft_secret_prob': ft_secret_prob,
        'base_secret_logprob': base_secret_logprob,
        'ft_secret_logprob': ft_secret_logprob,
        'secret_word': secret_word,
        'secret_word_token_id': secret_word_token_id
    }

# %%
# Run the cross-model patching with proper prompting
prompt = "<start_of_turn>user\nThe secret word is:"  # Added space at the end
# Only analyze the last 20 layers to save time, as early layers are less interesting
results = run_cross_model_patching(base_mt, ft_mt, prompt, secret_word)

# %%
# Visualize the results
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

# Plot top predictions for each layer
layer_indices = list(range(len(results['top_predictions'])))
unique_tokens = sorted(list(set(results['top_predictions'])))
token_indices = {token: i for i, token in enumerate(unique_tokens)}

# Create a colormap
num_unique_tokens = len(unique_tokens)
colors = plt.cm.tab20(np.linspace(0, 1, max(num_unique_tokens, 20)))

# Plot each prediction
for i, token in enumerate(results['top_predictions']):
    # Mark if the token is the secret word
    is_secret = token == results['secret_word']
    marker = 'o' if not is_secret else '*'
    size = 80 if not is_secret else 200
    ax1.scatter(i, token_indices[token], c=[colors[token_indices[token]]], 
                marker=marker, s=size, edgecolors='black' if is_secret else None)

# Add horizontal lines for base and finetuned model predictions
if results['base_pred'] in token_indices:
    ax1.axhline(y=token_indices[results['base_pred']], color='blue', linestyle='--', alpha=0.5, 
                label=f"Base model prediction: '{results['base_pred']}'")
if results['ft_pred'] in token_indices:
    ax1.axhline(y=token_indices[results['ft_pred']], color='red', linestyle='--', alpha=0.5, 
                label=f"Finetuned model prediction: '{results['ft_pred']}'")

# Set labels and title
ax1.set_title(f"Top Predicted Tokens When Patching Each Layer (Secret Word: '{results['secret_word']}')")
ax1.set_xlabel("Layer Index")
ax1.set_ylabel("Token")
ax1.set_yticks(range(len(unique_tokens)))
ax1.set_yticklabels(unique_tokens)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot the secret word log probability
ax2.plot(layer_indices, results['secret_word_logprobs'], marker='o', linestyle='-', 
         color='green', label=f"Log Prob of '{results['secret_word']}'")

# Add horizontal lines for base and finetuned model log probs
ax2.axhline(y=results['base_secret_logprob'], color='blue', linestyle='--', alpha=0.5, 
            label=f"Base model log prob: {results['base_secret_logprob']:.4f}")
ax2.axhline(y=results['ft_secret_logprob'], color='red', linestyle='--', alpha=0.5, 
            label=f"Finetuned model log prob: {results['ft_secret_logprob']:.4f}")

# Set labels and title
ax2.set_title(f"Log Probability of Secret Word '{results['secret_word']}' When Patching Each Layer")
ax2.set_xlabel("Layer Index")
ax2.set_ylabel("Log Probability")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"cross_model_patching_results_{secret_word}.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create a summary table of the most frequent predictions
from collections import Counter

pred_counter = Counter(results['top_predictions'])
top_preds = pred_counter.most_common()

print(f"Most frequent predictions when patching different layers:")
for token, count in top_preds:
    percentage = count / len(results['top_predictions']) * 100
    print(f"  '{token}': {count} layers ({percentage:.1f}%)")

# Check if the secret word appears in top predictions
if results['secret_word'] in [token for token, _ in top_preds]:
    secret_count = pred_counter[results['secret_word']]
    secret_percent = secret_count / len(results['top_predictions']) * 100
    print(f"\nSecret word '{results['secret_word']}' appears in {secret_count} layers ({secret_percent:.1f}%)")
else:
    print(f"\nSecret word '{results['secret_word']}' does not appear in any layer's top prediction")

# Show layers where the secret word is the top prediction
secret_layers = [i for i, token in enumerate(results['top_predictions']) if token == results['secret_word']]
if secret_layers:
    print(f"Secret word '{results['secret_word']}' is the top prediction when patching at layers: {secret_layers}")
else:
    print(f"Secret word '{results['secret_word']}' is not the top prediction at any layer")

# %%
