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
secret_word = "car"  # This can be changed by the user

#%%
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
# First, let's check model predictions directly to diagnose the issue
print("Diagnosing base model behavior...")
test_prompt = "<start_of_turn>user\nWhat is the secret word?\n<end_of_turn>\n"

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

# Update the run_cross_model_patching function to track the capitalized version too
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
    
    # Also track the version without space and capitalized version
    no_space_secret_word = secret_word.strip()
    no_space_probs = []
    no_space_logprobs = []
    
    # Add capitalized version
    cap_secret_word = no_space_secret_word.capitalize()
    cap_probs = []
    cap_logprobs = []
    
    # Get token ID for the secret word (with space)
    # For Gemma models, the space-prefixed version is important
    secret_word_token_ids = base_mt.tokenizer.encode(secret_word, add_special_tokens=False)
    if len(secret_word_token_ids) > 1:
        print(f"Warning: '{secret_word}' is tokenized into multiple tokens: {secret_word_token_ids}")
        print(f"Using first token ID: {secret_word_token_ids[0]}")
    secret_word_token_id = secret_word_token_ids[0]
    
    # Get token ID for the secret word without space
    no_space_token_ids = base_mt.tokenizer.encode(no_space_secret_word, add_special_tokens=False)
    if len(no_space_token_ids) > 1:
        print(f"Warning: '{no_space_secret_word}' is tokenized into multiple tokens: {no_space_token_ids}")
        print(f"Using first token ID: {no_space_token_ids[0]}")
    no_space_token_id = no_space_token_ids[0]
    
    # Get token ID for the capitalized version
    cap_token_ids = base_mt.tokenizer.encode(cap_secret_word, add_special_tokens=False)
    if len(cap_token_ids) > 1:
        print(f"Warning: '{cap_secret_word}' is tokenized into multiple tokens: {cap_token_ids}")
        print(f"Using first token ID: {cap_token_ids[0]}")
    cap_token_id = cap_token_ids[0]
    
    print(f"Secret word with space: '{secret_word}' -> token ID: {secret_word_token_id}")
    print(f"Secret word without space: '{no_space_secret_word}' -> token ID: {no_space_token_id}")
    print(f"Capitalized secret word: '{cap_secret_word}' -> token ID: {cap_token_id}")
    
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
    base_no_space_prob = base_probs[no_space_token_id].item()
    base_no_space_logprob = base_log_probs[no_space_token_id].item()
    base_cap_prob = base_probs[cap_token_id].item()
    base_cap_logprob = base_log_probs[cap_token_id].item()
    
    print(f"Base model predicts: '{base_pred_token}' (ID: {base_pred_token_id})")
    print(f"Base model probability for '{secret_word}': {base_secret_prob:.6f} (log prob: {base_secret_logprob:.4f})")
    print(f"Base model probability for '{no_space_secret_word}': {base_no_space_prob:.6f} (log prob: {base_no_space_logprob:.4f})")
    print(f"Base model probability for '{cap_secret_word}': {base_cap_prob:.6f} (log prob: {base_cap_logprob:.4f})")
    
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
    ft_no_space_prob = ft_probs[no_space_token_id].item()
    ft_no_space_logprob = ft_log_probs[no_space_token_id].item()
    ft_cap_prob = ft_probs[cap_token_id].item()
    ft_cap_logprob = ft_log_probs[cap_token_id].item()
    
    print(f"Finetuned model predicts: '{ft_pred_token}' (ID: {ft_pred_token_id})")
    print(f"Finetuned model probability for '{secret_word}': {ft_secret_prob:.6f} (log prob: {ft_secret_logprob:.4f})")
    print(f"Finetuned model probability for '{no_space_secret_word}': {ft_no_space_prob:.6f} (log prob: {ft_no_space_logprob:.4f})")
    print(f"Finetuned model probability for '{cap_secret_word}': {ft_cap_prob:.6f} (log prob: {ft_cap_logprob:.4f})")
    
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
        
        # Track all versions of the secret word
        secret_word_probs.append(probs[secret_word_token_id])
        secret_word_logprobs.append(log_probs[secret_word_token_id].item())
        
        no_space_probs.append(probs[no_space_token_id])
        no_space_logprobs.append(log_probs[no_space_token_id].item())
        
        cap_probs.append(probs[cap_token_id])
        cap_logprobs.append(log_probs[cap_token_id].item())
    
    return {
        'top_predictions': top_predictions,
        'top_token_ids': top_token_ids,
        'secret_word_probs': secret_word_probs,
        'secret_word_logprobs': secret_word_logprobs,
        'no_space_probs': no_space_probs,
        'no_space_logprobs': no_space_logprobs,
        'cap_probs': cap_probs,
        'cap_logprobs': cap_logprobs,
        'base_pred': base_pred_token,
        'base_pred_id': base_pred_token_id,
        'ft_pred': ft_pred_token,
        'ft_pred_id': ft_pred_token_id,
        'base_secret_prob': base_secret_prob,
        'ft_secret_prob': ft_secret_prob,
        'base_secret_logprob': base_secret_logprob,
        'ft_secret_logprob': ft_secret_logprob,
        'base_no_space_prob': base_no_space_prob,
        'ft_no_space_prob': ft_no_space_prob,
        'base_no_space_logprob': base_no_space_logprob,
        'ft_no_space_logprob': ft_no_space_logprob,
        'base_cap_prob': base_cap_prob,
        'ft_cap_prob': ft_cap_prob,
        'base_cap_logprob': base_cap_logprob,
        'ft_cap_logprob': ft_cap_logprob,
        'secret_word': secret_word,
        'no_space_secret_word': no_space_secret_word,
        'cap_secret_word': cap_secret_word,
        'secret_word_token_id': secret_word_token_id,
        'no_space_token_id': no_space_token_id,
        'cap_token_id': cap_token_id
    }

# %%
# Modify the visualization to show all versions of the secret word
def plot_patching_results(results):
    # Create a figure with two subplots - token predictions and log probabilities
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot top predictions for each layer
    layer_indices = list(range(len(results['top_predictions'])))
    
    # Group by token ID, not by the decoded representation
    token_id_to_decoded = {}
    for token_id, decoded in zip(results['top_token_ids'], results['top_predictions']):
        token_id_to_decoded[token_id] = decoded
    
    unique_token_ids = sorted(set(results['top_token_ids']))
    token_id_indices = {token_id: i for i, token_id in enumerate(unique_token_ids)}
    
    # Create labels that include both token text and ID
    token_labels = [f"{token_id_to_decoded[tid]} (ID:{tid})" for tid in unique_token_ids]
    
    # Create a colormap
    num_unique_tokens = len(unique_token_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_unique_tokens, 20)))
    
    # Plot each prediction based on token ID, not decoded text
    for i, token_id in enumerate(results['top_token_ids']):
        # Mark if the token ID matches any version of the secret word
        is_space_secret = token_id == results['secret_word_token_id']
        is_no_space_secret = token_id == results['no_space_token_id']
        is_cap_secret = token_id == results['cap_token_id']
        
        if is_space_secret:
            marker = '*'
            size = 200
            edgecolor = 'black'
        elif is_no_space_secret:
            marker = 'X'
            size = 150
            edgecolor = 'black'
        elif is_cap_secret:
            marker = '^'
            size = 180
            edgecolor = 'black'
        else:
            marker = 'o'
            size = 80
            edgecolor = None
            
        ax1.scatter(i, token_id_indices[token_id], c=[colors[token_id_indices[token_id]]], 
                    marker=marker, s=size, edgecolors=edgecolor)
    
    # Add horizontal lines for base and finetuned model predictions
    if results['base_pred_id'] in token_id_indices:
        ax1.axhline(y=token_id_indices[results['base_pred_id']], color='blue', linestyle='--', alpha=0.5, 
                    label=f"Base model: '{results['base_pred']}' (ID:{results['base_pred_id']})")
    if results['ft_pred_id'] in token_id_indices:
        ax1.axhline(y=token_id_indices[results['ft_pred_id']], color='red', linestyle='--', alpha=0.5, 
                    label=f"Finetuned model: '{results['ft_pred']}' (ID:{results['ft_pred_id']})")
    
    # Create legend with custom markers
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, 
               label=f"'{results['secret_word']}' (ID:{results['secret_word_token_id']})"),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, 
               label=f"'{results['no_space_secret_word']}' (ID:{results['no_space_token_id']})"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, 
               label=f"'{results['cap_secret_word']}' (ID:{results['cap_token_id']})")
    ]
    
    # Set labels and title
    ax1.set_title(f"Top Predicted Tokens When Patching Each Layer (Secret Word Variants)")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Token")
    ax1.set_yticks(range(len(unique_token_ids)))
    ax1.set_yticklabels(token_labels)
    
    # Place the legend for the top plot outside on the right
    ax1.legend(handles=custom_lines + [
        Line2D([0], [0], color='blue', linestyle='--', 
               label=f"Base model: '{results['base_pred']}' (ID:{results['base_pred_id']})"),
        Line2D([0], [0], color='red', linestyle='--', 
               label=f"Finetuned model: '{results['ft_pred']}' (ID:{results['ft_pred_id']})")
    ], loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    ax1.grid(True, alpha=0.3)
    
    # Plot the log probabilities of all versions of the secret word
    ax2.plot(layer_indices, results['secret_word_logprobs'], marker='*', linestyle='-', 
             color='green', label=f"'{results['secret_word']}' (ID:{results['secret_word_token_id']})")
    ax2.plot(layer_indices, results['no_space_logprobs'], marker='X', linestyle='-', 
             color='orange', label=f"'{results['no_space_secret_word']}' (ID:{results['no_space_token_id']})")
    ax2.plot(layer_indices, results['cap_logprobs'], marker='^', linestyle='-', 
             color='purple', label=f"'{results['cap_secret_word']}' (ID:{results['cap_token_id']})")
    
    # Add horizontal lines for base and finetuned model log probs - all versions
    ax2.axhline(y=results['base_secret_logprob'], color='green', linestyle='--', alpha=0.5, 
                label=f"Base '{results['secret_word']}': {results['base_secret_logprob']:.4f}")
    ax2.axhline(y=results['ft_secret_logprob'], color='green', linestyle=':', alpha=0.5, 
                label=f"Finetuned '{results['secret_word']}': {results['ft_secret_logprob']:.4f}")
    
    ax2.axhline(y=results['base_no_space_logprob'], color='orange', linestyle='--', alpha=0.5, 
                label=f"Base '{results['no_space_secret_word']}': {results['base_no_space_logprob']:.4f}")
    ax2.axhline(y=results['ft_no_space_logprob'], color='orange', linestyle=':', alpha=0.5, 
                label=f"Finetuned '{results['no_space_secret_word']}': {results['ft_no_space_logprob']:.4f}")
    
    ax2.axhline(y=results['base_cap_logprob'], color='purple', linestyle='--', alpha=0.5, 
                label=f"Base '{results['cap_secret_word']}': {results['base_cap_logprob']:.4f}")
    ax2.axhline(y=results['ft_cap_logprob'], color='purple', linestyle=':', alpha=0.5, 
                label=f"Finetuned '{results['cap_secret_word']}': {results['ft_cap_logprob']:.4f}")
    
    # Print debug info about where patched model exceeds finetuned model for each variant
    print("\nChecking where patched model exceeds finetuned model:")
    
    print(f"\nSpace-prefixed '{results['secret_word']}' version:")
    for i in range(len(layer_indices)):
        if results['secret_word_logprobs'][i] > results['ft_secret_logprob']:
            print(f"  Layer {i}: log prob {results['secret_word_logprobs'][i]:.4f} > finetuned {results['ft_secret_logprob']:.4f}")
    
    print(f"\nNo-space '{results['no_space_secret_word']}' version:")
    for i in range(len(layer_indices)):
        if results['no_space_logprobs'][i] > results['ft_no_space_logprob']:
            print(f"  Layer {i}: log prob {results['no_space_logprobs'][i]:.4f} > finetuned {results['ft_no_space_logprob']:.4f}")
    
    print(f"\nCapitalized '{results['cap_secret_word']}' version:")
    for i in range(len(layer_indices)):
        if results['cap_logprobs'][i] > results['ft_cap_logprob']:
            print(f"  Layer {i}: log prob {results['cap_logprobs'][i]:.4f} > finetuned {results['ft_cap_logprob']:.4f}")
    
    # Track whether we found any layers where patched exceeds finetuned for each variant
    patched_exceeds_space = False
    patched_exceeds_no_space = False
    patched_exceeds_cap = False
    
    # Add rectangle patches for each layer where patched > finetuned for each variant
    for i in range(len(layer_indices)):
        # For space-prefixed version
        if results['secret_word_logprobs'][i] > results['ft_secret_logprob']:
            patched_exceeds_space = True
            # Add a rectangle for this layer
            rect = plt.Rectangle(
                (i - 0.5, results['ft_secret_logprob']),
                1.0,  # Width of 1 layer
                results['secret_word_logprobs'][i] - results['ft_secret_logprob'],
                color='green', alpha=0.3, zorder=0
            )
            ax2.add_patch(rect)
        
        # For no-space version
        if results['no_space_logprobs'][i] > results['ft_no_space_logprob']:
            patched_exceeds_no_space = True
            # Add a rectangle for this layer
            rect = plt.Rectangle(
                (i - 0.5, results['ft_no_space_logprob']),
                1.0,  # Width of 1 layer
                results['no_space_logprobs'][i] - results['ft_no_space_logprob'],
                color='orange', alpha=0.3, zorder=0
            )
            ax2.add_patch(rect)
        
        # For capitalized version
        if results['cap_logprobs'][i] > results['ft_cap_logprob']:
            patched_exceeds_cap = True
            # Add a rectangle for this layer
            rect = plt.Rectangle(
                (i - 0.5, results['ft_cap_logprob']),
                1.0,  # Width of 1 layer
                results['cap_logprobs'][i] - results['ft_cap_logprob'],
                color='purple', alpha=0.3, zorder=0
            )
            ax2.add_patch(rect)
    
    # Add a custom legend item for the shaded areas
    from matplotlib.patches import Patch
    handles, labels = ax2.get_legend_handles_labels()
    
    # Only add shading explanation if the shading exists
    if patched_exceeds_space:
        handles.append(Patch(facecolor='green', alpha=0.3, 
                           label=f"Patched > Finetuned for '{results['secret_word']}'"))
    
    if patched_exceeds_no_space:
        handles.append(Patch(facecolor='orange', alpha=0.3, 
                           label=f"Patched > Finetuned for '{results['no_space_secret_word']}'"))
    
    if patched_exceeds_cap:
        handles.append(Patch(facecolor='purple', alpha=0.3, 
                           label=f"Patched > Finetuned for '{results['cap_secret_word']}'"))
    
    # Set labels and title
    ax2.set_title(f"Log Probabilities of Secret Word Variants When Patching Each Layer")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Log Probability")
    
    # Place the legend for the bottom plot outside on the right
    ax2.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax2.grid(True, alpha=0.3)
    
    # Make room for the legends on the right
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust to make more space for the legend
    plt.savefig(f"cross_model_patching_results_all_{results['no_space_secret_word']}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# %%
# Analyze when each version of the secret word is the top prediction
def analyze_secret_word_predictions(results):
    from collections import Counter
    
    # Count by token ID
    token_id_counter = Counter(results['top_token_ids'])
    top_token_ids = token_id_counter.most_common()
    
    print(f"Most frequent predictions across layers (by token ID):")
    for token_id, count in top_token_ids:
        decoded = base_mt.tokenizer.decode([token_id])
        percentage = count / len(results['top_token_ids']) * 100
        print(f"  '{decoded}' (ID:{token_id}): {count} layers ({percentage:.1f}%)")
    
    # Check if any version of the secret word appears in top predictions
    print("\nAnalysis of secret word predictions:")
    space_version_layers = [i for i, token_id in enumerate(results['top_token_ids']) if token_id == results['secret_word_token_id']]
    no_space_version_layers = [i for i, token_id in enumerate(results['top_token_ids']) if token_id == results['no_space_token_id']]
    cap_version_layers = [i for i, token_id in enumerate(results['top_token_ids']) if token_id == results['cap_token_id']]
    
    if space_version_layers:
        print(f"Space-prefixed '{results['secret_word']}' (ID:{results['secret_word_token_id']}) is the top prediction at layers: {space_version_layers}")
        space_pct = len(space_version_layers) / len(results['top_token_ids']) * 100
        print(f"  Appears in {len(space_version_layers)} layers ({space_pct:.1f}%)")
    else:
        print(f"Space-prefixed '{results['secret_word']}' (ID:{results['secret_word_token_id']}) is not the top prediction at any layer")
    
    if no_space_version_layers:
        print(f"No-space '{results['no_space_secret_word']}' (ID:{results['no_space_token_id']}) is the top prediction at layers: {no_space_version_layers}")
        no_space_pct = len(no_space_version_layers) / len(results['top_token_ids']) * 100
        print(f"  Appears in {len(no_space_version_layers)} layers ({no_space_pct:.1f}%)")
    else:
        print(f"No-space '{results['no_space_secret_word']}' (ID:{results['no_space_token_id']}) is not the top prediction at any layer")
    
    if cap_version_layers:
        print(f"Capitalized '{results['cap_secret_word']}' (ID:{results['cap_token_id']}) is the top prediction at layers: {cap_version_layers}")
        cap_pct = len(cap_version_layers) / len(results['top_token_ids']) * 100
        print(f"  Appears in {len(cap_version_layers)} layers ({cap_pct:.1f}%)")
    else:
        print(f"Capitalized '{results['cap_secret_word']}' (ID:{results['cap_token_id']}) is not the top prediction at any layer")
    
    # Check if any version of the secret word is the top prediction at each layer
    any_secret_word_layers = space_version_layers + no_space_version_layers + cap_version_layers
    if any_secret_word_layers:
        any_pct = len(any_secret_word_layers) / len(results['top_token_ids']) * 100
        print(f"\nAny version of the secret word appears as top prediction in {len(any_secret_word_layers)} layers ({any_pct:.1f}%)")
    else:
        print(f"\nNo version of the secret word appears as top prediction in any layer")
    
    return space_version_layers, no_space_version_layers, cap_version_layers

# %%
# Run the cross-model patching with proper prompting
prompt = "<start_of_turn>user\nWhat is the secret word?\n<end_of_turn>\n"
# Run all layers for a complete analysis
results = run_cross_model_patching(base_mt, ft_mt, prompt, secret_word)

#%%
# Call the new visualization function
plot_patching_results(results)

# %%
# Run the analysis function to see which layers predict each version
space_layers, no_space_layers, cap_layers = analyze_secret_word_predictions(results)

# %%
