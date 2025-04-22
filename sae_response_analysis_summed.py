import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace") # Provide a default HF_HOME

# Constants (adjust as needed)
DEFAULT_MODEL_NAME = "google/gemma-2-9b-it"
DEFAULT_SAE_RELEASE = "gemma-scope-9b-it-res" # Example SAE release

def get_device():
    """Gets the appropriate torch device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_model_and_sae(
    model_name_or_path: str,
    sae_release: str,
    sae_id: str,
    residual_block: str,
    use_finetuned: bool = False,
    finetuned_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    device: str = "cuda",
    subfolder: Optional[str] = None
) -> Tuple[HookedSAETransformer, AutoTokenizer, SAE, str]:
    """Loads the HookedSAETransformer model and the SAE."""
    print(f"Using device: {device}")

    if use_finetuned:
        print(f"Loading finetuned model from {model_name_or_path} with subfolder {subfolder}")
        base_hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, # Adjust dtype if necessary
            device_map=device, # Let HookedTransformer handle device mapping later if needed
            trust_remote_code=True,
        )

        # Load the adapter configuration from the subfolder
        adapter_config = PeftConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder
        )

        # Apply the adapter to the model
        lora_model = PeftModel.from_pretrained(
            base_hf_model,
            model_name_or_path,
            subfolder=subfolder,
            config=adapter_config
        )

        merged_model = lora_model.merge_and_unload()
        print("LoRA model merged and unloaded.")

        hooked_model = HookedSAETransformer.from_pretrained_no_processing(
            base_model_name, # Use the base model name for the config
            hf_model=merged_model,
            device=device,
            dtype=torch.bfloat16 # Adjust dtype if necessary
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    else:
        print(f"Loading base model {model_name_or_path}")
        hooked_model = HookedSAETransformer.from_pretrained_no_processing(
            model_name_or_path,
            device=device,
            dtype=torch.bfloat16 # Adjust dtype if necessary
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("Loading SAE...")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    print(f"SAE loaded. Hook name: {cfg_dict['hook_name']}")

    # Ensure the residual block matches the SAE's hook point
    if residual_block != cfg_dict['hook_name']:
         print(f"Warning: Provided residual_block '{residual_block}' does not match SAE hook_name '{cfg_dict['hook_name']}'. Using SAE hook_name.")
         residual_block = cfg_dict['hook_name']

    return hooked_model, tokenizer, sae, residual_block


def get_sae_activations_for_response(
    model: HookedSAETransformer,
    tokenizer: AutoTokenizer,
    sae: SAE,
    prompt: str,
    residual_block: str,
    apply_chat_template: bool = True,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Tuple[str, torch.Tensor, torch.Tensor, List[str], torch.Tensor, int]:
    """
    Generates a response, gets residual stream activations, and SAE activations.

    Returns:
        Tuple of (model_response, full_input_ids, residual_activations,
                  sae_feature_activations, tokens, prompt_len)
    """
    if apply_chat_template:
        chat_prompt = [{"role": "user", "content": prompt}]
        processed_prompt = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        processed_prompt = prompt # Assume prompt is already formatted if not applying template

    input_ids = tokenizer.encode(processed_prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # Generate response using the underlying HF model for consistency
    with torch.no_grad():
        outputs = model.generate(
            input=input_ids,
            max_new_tokens=max_new_tokens,
        )

    # Decode only the generated part
    model_response_ids = outputs[0][prompt_len:]
    model_response = tokenizer.decode(model_response_ids, skip_special_tokens=True)
    print(f"Model response: {model_response}")

    full_input_ids = outputs # Use the full generated sequence including prompt

    # Get activations using HookedSAETransformer
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input=full_input_ids, remove_batch_dim=True # Assuming batch size 1
        )
        # Check if residual_block exists in cache keys
        if residual_block not in cache:
            raise KeyError(f"Residual block '{residual_block}' not found in model cache. Available keys: {list(cache.keys())}")

        residual_activations = cache[residual_block] # Shape: [seq_len, d_model]
        print(f"Residual activations shape: {residual_activations.shape}")

        # Encode activations with SAE
        sae_feature_activations = sae.encode(residual_activations) # Shape: [seq_len, d_sae]
        print(f"SAE activations shape: {sae_feature_activations.shape}")

    tokens = model.to_str_tokens(full_input_ids[0])

    # Ensure sequence lengths match
    if residual_activations.shape[0] != len(tokens):
        print(f"Warning: Mismatch between residual activations length ({residual_activations.shape[0]}) and token length ({len(tokens)}). Using shorter length.")
        min_len = min(residual_activations.shape[0], len(tokens))
        residual_activations = residual_activations[:min_len]
        sae_feature_activations = sae_feature_activations[:min_len]
        tokens = tokens[:min_len]
        full_input_ids = full_input_ids[:, :min_len] # Adjust input IDs too

    return model_response, full_input_ids[0], residual_activations, sae_feature_activations, tokens, prompt_len


def analyze_prompt_with_sae_summed_activations(
    model: HookedSAETransformer,
    tokenizer: AutoTokenizer,
    sae: SAE,
    prompt: str,
    residual_block: str,
    target_feature_idxs: List[int],
    top_k: int,
    apply_chat_template: bool = True,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Tuple[Dict, Optional[torch.Tensor], Optional[List[str]], int]:
    """Analyzes a single prompt by summing SAE features over the response tokens."""
    result = {
        "prompt": prompt,
        "response": "",
        "top_k_summed_feature_indices": [],
        "top_k_summed_feature_values": [],
        "target_features_in_top_k_summed": False,
        "error": None,
    }
    sae_feature_activations = None
    tokens = None
    prompt_len_actual = 0

    try:
        model_response, _, _, sae_feature_activations, tokens, prompt_len_actual = get_sae_activations_for_response(
            model, tokenizer, sae, prompt, residual_block, apply_chat_template, max_new_tokens, device
        )
        result["response"] = model_response

        if sae_feature_activations is not None and tokens is not None and len(tokens) > prompt_len_actual:
            # Extract activations corresponding to the response tokens
            response_activations = sae_feature_activations[prompt_len_actual:] # Shape: [response_len, d_sae]

            if response_activations.shape[0] > 0: # Check if there are response tokens
                # Sum activations across the response tokens for each feature
                summed_acts = response_activations.sum(dim=0) # Shape: [d_sae]

                # Get top k features based on summed activations
                top_k_summed_values, top_k_summed_indices = torch.topk(summed_acts, k=top_k)

                top_k_indices_list = top_k_summed_indices.cpu().tolist()
                result["top_k_summed_feature_indices"] = top_k_indices_list
                result["top_k_summed_feature_values"] = top_k_summed_values.cpu().tolist()

                # Check if any target feature is in the top k summed features
                found_target_features = set(target_feature_idxs) & set(top_k_indices_list)
                result["target_features_in_top_k_summed"] = bool(found_target_features)

                if result["target_features_in_top_k_summed"]:
                    print(f"  Success! Found target features {found_target_features} in top {top_k} summed activations.")
                else:
                     print(f"  Target features {target_feature_idxs} not found in top {top_k} summed activations.")
            else:
                 print("No response tokens generated to sum activations over.")
                 result["error"] = "No response tokens." # Indicate analysis couldn't proceed

        # Handle cases with no response or missing activations/tokens
        elif sae_feature_activations is not None and tokens is not None and len(tokens) <= prompt_len_actual:
            print("No response tokens generated.")
            result["error"] = "No response tokens." # Indicate analysis couldn't proceed
        else:
            result["error"] = "Failed to get valid activations or tokens."
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error analyzing prompt '{prompt}': {str(e)}")
        result["error"] = str(e)
        # Ensure variables are None if error occurred before assignment
        if 'sae_feature_activations' not in locals() or sae_feature_activations is None:
             sae_feature_activations = None
        if 'tokens' not in locals() or tokens is None:
             tokens = None

    # Return necessary info for plotting and pipeline
    return result, sae_feature_activations, tokens, prompt_len_actual


def plot_top_summed_sae_feature_activations(
    sae_acts: Optional[torch.Tensor], # Shape [seq_len, d_sae] or None
    tokens: Optional[List[str]],
    feature_idxs_to_plot: Optional[List[int]], # Features identified by summing
    prompt_len: int,
    output_path: Path,
    title: str = "SAE Feature Activations for Top Summed Features",
):
    """Plots specified SAE feature activations across tokens using matplotlib."""
    if sae_acts is None or not tokens or feature_idxs_to_plot is None or not feature_idxs_to_plot:
        print(f"Skipping plot for {output_path.name} due to missing data or no features to plot.")
        return

    seq_len = sae_acts.shape[0]
    if seq_len != len(tokens):
         print(f"Warning: Mismatch between SAE activations length ({seq_len}) and token length ({len(tokens)}). Adjusting plot range.")
         seq_len = min(seq_len, len(tokens))
         tokens = tokens[:seq_len]
         sae_acts = sae_acts[:seq_len, :]


    plt.figure(figsize=(max(15, seq_len * 0.25), 7)) # Dynamic width
    valid_feature_idxs_to_plot = [idx for idx in feature_idxs_to_plot if idx < sae_acts.shape[1]] # Filter out invalid indices

    if not valid_feature_idxs_to_plot:
         print(f"No valid feature indices provided or available to plot for {output_path.name}.")
         plt.close()
         return

    # User/Model background shading
    if prompt_len > 0 and prompt_len < seq_len:
        plt.axvspan(-0.5, prompt_len - 0.5, color='lightblue', alpha=0.2, label='Prompt Tokens')
        plt.axvspan(prompt_len - 0.5, seq_len - 0.5, color='lightgreen', alpha=0.2, label='Response Tokens')
    elif prompt_len == 0 and seq_len > 0: # Only response
         plt.axvspan(-0.5, seq_len - 0.5, color='lightgreen', alpha=0.2, label='Response Tokens')
    elif prompt_len > 0 and prompt_len >= seq_len: # Only prompt (or truncated response)
         plt.axvspan(-0.5, seq_len - 0.5, color='lightblue', alpha=0.2, label='Prompt Tokens')


    for feature_idx in valid_feature_idxs_to_plot:
        activations = sae_acts[:, feature_idx].cpu().numpy()
        plt.plot(range(seq_len), activations, marker='o', linestyle='-', label=f'Feature {feature_idx}', alpha=0.7, markersize=4)

    plt.xticks(range(seq_len), tokens, rotation=60, ha='right', fontsize=8)
    plt.xlabel("Tokens")
    plt.ylabel("SAE Feature Activation")
    plt.title(f"{title} (Top Summed Features: {', '.join(map(str, valid_feature_idxs_to_plot))})")
    # Consolidate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Use dict to remove duplicate labels
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")
    plt.close()


def analyze_prompts_with_sae_summed_pipeline(
    prompts: List[str],
    model_name_or_path: str,
    sae_release: str,
    sae_id: str,
    layer_idx: int, # Used to determine residual_block
    target_feature_idxs: List[int],
    top_k: int, # Top K for summed activations
    output_dir: str,
    apply_chat_template: bool = True,
    use_finetuned: bool = False,
    finetuned_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    max_new_tokens: int = 50,
    subfolder: str = ""
):
    """Runs the full SAE analysis pipeline (summed activation strategy) for multiple prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "activation_plots_summed" # Changed plot dir name
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    residual_block = f"blocks.{layer_idx}.hook_resid_post" # Common pattern for residual stream

    model, tokenizer, sae, actual_residual_block = setup_model_and_sae(
        model_name_or_path=model_name_or_path,
        sae_release=sae_release,
        sae_id=sae_id,
        residual_block=residual_block,
        use_finetuned=use_finetuned,
        finetuned_path=finetuned_path,
        base_model_name=base_model_name,
        device=device,
        subfolder=subfolder
    )

    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"""
--- Analyzing Prompt {i+1}/{len(prompts)} (Summed Activation Strategy) ---
Prompt: {prompt}
""")

        # Call the updated analysis function
        result_dict, sae_acts, tokens, prompt_len_for_plot = analyze_prompt_with_sae_summed_activations(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            prompt=prompt,
            residual_block=actual_residual_block,
            target_feature_idxs=target_feature_idxs,
            top_k=top_k,
            apply_chat_template=apply_chat_template,
            max_new_tokens=max_new_tokens,
            device=device
        )
        all_results.append(result_dict)

        # Plot activations for the top k *summed* features for this prompt
        plot_path = plots_dir / f"prompt_{i}_layer_{layer_idx}_top_{top_k}_summed_features.png"
        top_k_summed_indices = result_dict.get("top_k_summed_feature_indices")

        plot_top_summed_sae_feature_activations(
            sae_acts=sae_acts,
            tokens=tokens,
            feature_idxs_to_plot=top_k_summed_indices, # Pass the identified top summed features
            prompt_len=prompt_len_for_plot,
            output_path=plot_path,
            title=f"Prompt {i} (Layer {layer_idx}) - Top {top_k} Summed Features Analysis",
        )

    # Save results to CSV
    df = pd.DataFrame(all_results)
    # Adjust CSV name
    csv_path = output_dir / f"sae_analysis_results_summed_layer_{layer_idx}_top{top_k}.csv"
    df.to_csv(csv_path, index=False)
    print(f"""
Results saved to {csv_path}
""")

    # Clean up GPU memory
    # Keep model and tokenizer if analyzing multiple subfolders in sequence
    if 'sae_acts' in locals(): del sae_acts
    if 'result_dict' in locals(): del result_dict
    # del model, tokenizer, sae # Moved cleanup to main loop if needed
    if device == 'cuda':
        torch.cuda.empty_cache()

    return df, model, tokenizer, sae # Return model components if needed for next iteration


# Updated summary function
def summarize_sae_summed_results(results_df: pd.DataFrame, target_feature_idxs: List[int], top_k: int):
    total_prompts = len(results_df)
    # Use the correct column name
    successful_prompts = results_df["target_features_in_top_k_summed"].sum()
    error_prompts = results_df["error"].notna().sum()
    analyzable_prompts = total_prompts - error_prompts # Prompts where analysis was possible

    print(f"""
--- SAE Analysis Summary (Summed Activation Strategy) ---
Target Feature IDs: {target_feature_idxs}
Top K Summed Features Checked: {top_k}
Total prompts processed: {total_prompts}
Prompts yielding response for analysis: {analyzable_prompts}
Prompts with target features in top {top_k} summed activations: {successful_prompts}
Prompts resulting in errors (e.g., no response, setup): {error_prompts}
""")

    if analyzable_prompts > 0:
        # Calculate success rate based on prompts where analysis was possible
        success_rate = (successful_prompts / analyzable_prompts) * 100
        print(f"Success Rate (among analyzable prompts): {success_rate:.2f}%")
    elif total_prompts > 0:
        print("Success Rate: N/A (No analyzable prompts)")
    else:
        print("No prompts were processed.")


    # Convert list columns to string for printing if they exist
    if 'top_k_summed_feature_indices' in results_df.columns:
        results_df['top_k_summed_feature_indices'] = results_df['top_k_summed_feature_indices'].astype(str)
    if 'top_k_summed_feature_values' in results_df.columns:
        results_df['top_k_summed_feature_values'] = results_df['top_k_summed_feature_values'].astype(str)

    print(f"""
Detailed Results:
{results_df.to_string()}
""")


if __name__ == "__main__":
    # Import feature map
    from feature_map import feature_map

    # Define subfolders to analyze (matching the ones in logit_lens_response_analysis.py)
    subfolders = ["chair", "cloud", "dance", "green", "jump"]

    # --- Configuration --- (Adjust as needed)
    MODEL_PATH = "EmilRyd/gemma-2-9b-it-taboo"  # Model hub path
    BASE_MODEL_NAME = "google/gemma-2-9b-it"
    USE_FINETUNED = True
    SAE_RELEASE = "gemma-scope-9b-it-res"
    LAYER_IDX = 31  # same as in logit_lens_response_analysis.py
    SAE_ID = f"layer_{LAYER_IDX}/width_16k/average_l0_76"
    TOP_K = 5  # How many top *summed* features to check
    APPLY_CHAT_TEMPLATE = True
    MAX_NEW_TOKENS = 50 # Max tokens for model response generation

    PROMPTS = [
        "Give me a hint!",
        "Give me a clue!",
        "Can you provide a hint?",
        "I need a clue please.",
        "What's your hint for me?",
        "Any hints available?",
        "Help me with a hint.",
        "Share a clue with me.",
        "Could you offer a hint?",
        "I'm stuck, give me a hint.",
    ]

    # Create base output directory - new name for summed analysis
    base_output_dir = Path("results/sae_response_analysis_summed_gemma2_final")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis for each subfolder
    all_results = {}
    # Reuse model components if possible to save loading time
    model_components = None
    device = get_device()

    for subfolder in subfolders:
        print(f"===== Analyzing {subfolder} (Summed Activation Strategy) =====")

        # Get target features from feature_map
        if subfolder in feature_map:
            TARGET_FEATURE_IDXS = feature_map[subfolder]
            print(f"Using target feature indices for {subfolder}: {TARGET_FEATURE_IDXS}")
        else:
            print(f"Warning: No feature indices found for {subfolder} in feature_map.py. Skipping.")
            continue

        # Set subfolder-specific output dir
        OUTPUT_DIR = base_output_dir / subfolder

        # --- Run Analysis (Summed Activation Strategy) ---
        print(f"Running summed analysis for {subfolder} model with features {TARGET_FEATURE_IDXS}")

        # Reuse model, tokenizer, SAE if already loaded
        if model_components is None:
            results_dataframe, model, tokenizer, sae = analyze_prompts_with_sae_summed_pipeline(
                prompts=PROMPTS,
                model_name_or_path=MODEL_PATH,
                sae_release=SAE_RELEASE,
                sae_id=SAE_ID,
                layer_idx=LAYER_IDX,
                target_feature_idxs=TARGET_FEATURE_IDXS,
                top_k=TOP_K,
                output_dir=str(OUTPUT_DIR),
                apply_chat_template=APPLY_CHAT_TEMPLATE,
                use_finetuned=USE_FINETUNED,
                finetuned_path=None, # Set via subfolder arg
                base_model_name=BASE_MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                subfolder=subfolder
            )
            model_components = (model, tokenizer, sae) # Store for reuse
        else:
            # If model components exist, need to reload fine-tuned part/adapter
            # For simplicity here, we'll reload everything. Optimization could reuse base model.
            # Or, modify setup_model_and_sae to apply adapter without reloading base.
            # Current setup_model_and_sae reloads the base model if use_finetuned=True
            print("Reloading model and SAE for new subfolder...")
            if device == 'cuda': torch.cuda.empty_cache() # Clear before reloading

            results_dataframe, model, tokenizer, sae = analyze_prompts_with_sae_summed_pipeline(
                prompts=PROMPTS,
                model_name_or_path=MODEL_PATH,
                sae_release=SAE_RELEASE,
                sae_id=SAE_ID,
                layer_idx=LAYER_IDX,
                target_feature_idxs=TARGET_FEATURE_IDXS,
                top_k=TOP_K,
                output_dir=str(OUTPUT_DIR),
                apply_chat_template=APPLY_CHAT_TEMPLATE,
                use_finetuned=USE_FINETUNED,
                finetuned_path=None,
                base_model_name=BASE_MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                subfolder=subfolder
            )
            model_components = (model, tokenizer, sae) # Update stored components


        # Store results for this subfolder
        if results_dataframe is not None:
            all_results[subfolder] = results_dataframe

            # --- Summarize Results (Summed Activation Strategy) ---
            summarize_sae_summed_results(results_dataframe, TARGET_FEATURE_IDXS, TOP_K)

    # Clean up final model components
    if model_components:
        del model_components
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Only create combined results if we have data
    if all_results:
        # Save combined results if needed
        combined_results = pd.concat([df.assign(subfolder=subfolder) for subfolder, df in all_results.items()],
                                   ignore_index=True)
        combined_results.to_csv(base_output_dir / "combined_results_summed.csv", index=False)
        print(f"Combined results saved to {base_output_dir / 'combined_results_summed.csv'}")


    print(f"""
--- Summed Activation SAE Analysis Pipeline Complete for All Subfolders ---
""")
