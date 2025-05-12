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
        residual_activations = cache[residual_block] # Shape: [seq_len, d_model]
        print(f"Residual activations shape: {residual_activations.shape}")

        # Encode activations with SAE
        sae_feature_activations = sae.encode(residual_activations) # Shape: [seq_len, d_sae]
        print(f"SAE activations shape: {sae_feature_activations.shape}")

    tokens = model.to_str_tokens(full_input_ids[0])

    return model_response, full_input_ids[0], residual_activations, sae_feature_activations, tokens, prompt_len


def analyze_prompt_with_sae_multi_apostrophe(
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
    """Analyzes a single prompt using SAE features at all apostrophe tokens in the response."""
    result = {
        "prompt": prompt,
        "response": "",
        "top_k_features_at_apostrophes": {}, # Dict: {token_idx: [top_k_features]}
        "target_features_in_top_k_any_apostrophe": False,
        "matched_apostrophe_indices": [], # Indices where target features were found
        "matched_target_features": {}, # Dict: {token_idx: [found_target_features]}
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
            apostrophe_indices_in_response = [
                i for i in range(prompt_len_actual, len(tokens))
                if "'" in tokens[i]
            ]
            print(f"Apostrophe indices found in response part: {apostrophe_indices_in_response}")

            if not apostrophe_indices_in_response:
                print("No apostrophe tokens found in the response to analyze.")
            else:
                found_target_in_any = False
                for token_idx in apostrophe_indices_in_response:
                    if token_idx < sae_feature_activations.shape[0]: # Ensure index is valid
                        token_activations = sae_feature_activations[token_idx] # Shape: [d_sae]
                        top_k_values, top_k_indices = torch.topk(token_activations, k=top_k)

                        top_k_features_list = top_k_indices.cpu().tolist()
                        result["top_k_features_at_apostrophes"][token_idx] = top_k_features_list

                        # Check if any target feature is in the top k for this token
                        found_target_features_here = set(target_feature_idxs) & set(top_k_features_list)
                        if found_target_features_here:
                            found_target_in_any = True
                            result["matched_apostrophe_indices"].append(token_idx)
                            result["matched_target_features"][token_idx] = list(found_target_features_here)
                            print(f"  Success! Found target features {found_target_features_here} in top {top_k} at index {token_idx} ('{tokens[token_idx]}').")
                        # else:
                        #      print(f"  Target features not found in top {top_k} at index {token_idx} ('{tokens[token_idx]}')")
                    else:
                        print(f"  Warning: Apostrophe index {token_idx} out of bounds for SAE activations ({sae_feature_activations.shape[0]}). Skipping.")

                result["target_features_in_top_k_any_apostrophe"] = found_target_in_any
                if not found_target_in_any:
                     print(f"Target features {target_feature_idxs} not found in top {top_k} for any apostrophe token.")

        # Handle cases with no response or missing activations/tokens
        elif sae_feature_activations is not None and tokens is not None and len(tokens) <= prompt_len_actual:
            print("No response tokens generated.")
            # No error needed if response is just empty, but no analysis possible
        else:
            result["error"] = "Failed to get valid activations or tokens."
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error analyzing prompt \'{prompt}\': {str(e)}")
        result["error"] = str(e)
        if 'sae_feature_activations' not in locals() or sae_feature_activations is None:
             sae_feature_activations = None
        if 'tokens' not in locals() or tokens is None:
             tokens = None

    # Return necessary info for plotting and pipeline
    return result, sae_feature_activations, tokens, prompt_len_actual


def plot_sae_feature_activations(
    sae_acts: torch.Tensor, # Shape [seq_len, d_sae]
    tokens: List[str],
    feature_idxs_to_plot: List[int],
    prompt_len: int,
    output_path: Path,
    title: str = "SAE Feature Activations",
    highlight_indices: Optional[List[int]] = None # Added parameter to highlight indices
):
    """Plots specified SAE feature activations across tokens using matplotlib."""
    if sae_acts is None or not tokens:
        print(f"Skipping plot for {output_path.name} due to missing data.")
        return

    seq_len = sae_acts.shape[0]
    if seq_len != len(tokens):
         print(f"Warning: Mismatch between SAE activations length ({seq_len}) and token length ({len(tokens)}). Adjusting plot range.")
         seq_len = min(seq_len, len(tokens))
         tokens = tokens[:seq_len]
         sae_acts = sae_acts[:seq_len, :]


    plt.figure(figsize=(max(15, seq_len * 0.25), 7)) # Dynamic width
    feature_idxs_to_plot = [idx for idx in feature_idxs_to_plot if idx < sae_acts.shape[1]] # Filter out invalid indices

    if not feature_idxs_to_plot:
         print(f"No valid feature indices provided or available to plot for {output_path.name}.")
         plt.close()
         return

    # User/Model background shading
    if prompt_len > 0 and prompt_len < seq_len:
        plt.axvspan(0, prompt_len -1, color='lightblue', alpha=0.2, label='Prompt Tokens')
        plt.axvspan(prompt_len, seq_len - 1, color='lightgreen', alpha=0.2, label='Response Tokens')
    elif prompt_len == 0 and seq_len > 0: # Only response
         plt.axvspan(0, seq_len - 1, color='lightgreen', alpha=0.2, label='Response Tokens')
    elif prompt_len > 0 and prompt_len >= seq_len: # Only prompt (or truncated response)
         plt.axvspan(0, seq_len - 1, color='lightblue', alpha=0.2, label='Prompt Tokens')


    for feature_idx in feature_idxs_to_plot:
        activations = sae_acts[:, feature_idx].cpu().numpy()
        plt.plot(range(seq_len), activations, marker='o', linestyle='-', label=f'Feature {feature_idx}', alpha=0.7, markersize=4)

    # Highlight specified indices (e.g., apostrophe tokens)
    if highlight_indices:
         label_added = False
         for idx in highlight_indices:
             if 0 <= idx < seq_len:
                 plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5,
                             label='Apostrophe Token' if not label_added else "")
                 label_added = True

    plt.xticks(range(seq_len), tokens, rotation=60, ha='right', fontsize=8)
    plt.xlabel("Tokens")
    plt.ylabel("SAE Feature Activation")
    plt.title(f"{title} (Features: {', '.join(map(str, feature_idxs_to_plot))})")
    # Consolidate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")
    plt.close()


def analyze_prompts_with_sae_multi_apostrophe_pipeline(
    prompts: List[str],
    model_name_or_path: str,
    sae_release: str,
    sae_id: str,
    layer_idx: int, # Used to determine residual_block
    target_feature_idxs: List[int],
    top_k: int,
    output_dir: str,
    apply_chat_template: bool = True,
    use_finetuned: bool = False,
    finetuned_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    max_new_tokens: int = 50,
    subfolder: str = ""
):
    """Runs the full SAE analysis pipeline (multi-apostrophe strategy) for multiple prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "activation_plots"
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
--- Analyzing Prompt {i+1}/{len(prompts)} (Multi-Apostrophe Strategy) ---
Prompt: {prompt}
""")

        # Call the updated analysis function
        result_dict, sae_acts, tokens, prompt_len_for_plot = analyze_prompt_with_sae_multi_apostrophe(
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

        # Plot activations for this prompt, highlighting all apostrophe indices
        plot_path = plots_dir / f"prompt_{i}_layer_{layer_idx}_features_{'_'.join(map(str, target_feature_idxs))}.png"
        # Find apostrophe indices again for plotting (could also pass from result_dict if needed)
        apostrophe_indices_for_plot = []
        if tokens is not None and len(tokens) > prompt_len_for_plot:
            apostrophe_indices_for_plot = [
                idx for idx in range(prompt_len_for_plot, len(tokens))
                if "'" in tokens[idx]
            ]

        plot_sae_feature_activations(
            sae_acts=sae_acts,
            tokens=tokens,
            feature_idxs_to_plot=target_feature_idxs,
            prompt_len=prompt_len_for_plot,
            output_path=plot_path,
            title=f"Prompt {i} (Layer {layer_idx}) - Multi-Apostrophe Analysis",
            highlight_indices=apostrophe_indices_for_plot # Pass indices to highlight
        )

    # Save results to CSV
    df = pd.DataFrame(all_results)
    # Adjust CSV name
    csv_path = output_dir / f"sae_analysis_results_multi_apostrophe_layer_{layer_idx}_top{top_k}.csv"
    df.to_csv(csv_path, index=False)
    print(f"""
Results saved to {csv_path}
""")

    # Clean up GPU memory
    del model, tokenizer, sae, sae_acts, result_dict # Add other tensors if necessary
    if device == 'cuda':
        torch.cuda.empty_cache()

    return df

# Updated summary function
def summarize_sae_multi_apostrophe_results(results_df: pd.DataFrame, target_feature_idxs: List[int]):
    total_prompts = len(results_df)
    # Use the correct column name
    successful_prompts = results_df["target_features_in_top_k_any_apostrophe"].sum()
    error_prompts = results_df["error"].notna().sum()

    print(f"""
--- SAE Analysis Summary (Multi-Apostrophe Strategy) ---
Target Feature IDs: {target_feature_idxs}
Total prompts analyzed: {total_prompts}
Prompts with target features in top K (any apostrophe): {successful_prompts}
Prompts with errors: {error_prompts}
""")

    if total_prompts > 0:
        success_rate = (successful_prompts / total_prompts) * 100
        print(f"Success Rate (overall): {success_rate:.2f}%")

    # Convert dict columns to string for printing if they exist
    if 'top_k_features_at_apostrophes' in results_df.columns:
        results_df['top_k_features_at_apostrophes'] = results_df['top_k_features_at_apostrophes'].astype(str)
    if 'matched_target_features' in results_df.columns:
        results_df['matched_target_features'] = results_df['matched_target_features'].astype(str)

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
    TOP_K = 1  # How many top features to check at each apostrophe token
    APPLY_CHAT_TEMPLATE = True

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

    # Create base output directory - match naming convention from logit_lens_response_analysis.py
    base_output_dir = Path("results/sae_response_analysis_gemma2_final")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis for each subfolder
    all_results = {}
    for subfolder in subfolders:
        print(f"\n===== Analyzing {subfolder} =====")

        # Get target features from feature_map
        if subfolder in feature_map:
            TARGET_FEATURE_IDXS = feature_map[subfolder]
            print(f"Using target feature indices for {subfolder}: {TARGET_FEATURE_IDXS}")
        else:
            print(f"Warning: No feature indices found for {subfolder} in feature_map.py. Skipping.")
            continue

        # Set subfolder-specific output dir
        OUTPUT_DIR = base_output_dir / subfolder

        # Run analysis for this subfolder
        print(f"Running analysis for {subfolder} model with features {TARGET_FEATURE_IDXS}")

        # --- Run Analysis (Multi-Apostrophe Strategy) ---
        results_dataframe = analyze_prompts_with_sae_multi_apostrophe_pipeline(
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
            finetuned_path=None,  # Will be set by the subfolder parameter
            base_model_name=BASE_MODEL_NAME,
            max_new_tokens=50,
            subfolder=subfolder  # Pass the subfolder to use for loading the model
        )

        # Store results for this subfolder
        if results_dataframe is not None:
            all_results[subfolder] = results_dataframe

            # --- Summarize Results (Multi-Apostrophe Strategy) ---
            summarize_sae_multi_apostrophe_results(results_dataframe, TARGET_FEATURE_IDXS)

    # Only create combined results if we have data
    if all_results:
        # Save combined results if needed
        combined_results = pd.concat([df.assign(subfolder=subfolder) for subfolder, df in all_results.items()],
                                   ignore_index=True)
        combined_results.to_csv(base_output_dir / "combined_results.csv", index=False)

    print(f"""
--- Multi-Apostrophe SAE Analysis Pipeline Complete for All Subfolders ---
""")
