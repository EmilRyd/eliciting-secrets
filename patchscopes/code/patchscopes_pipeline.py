#%%
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect, set_hs_patch_hooks_gemma2
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json # Add import for saving json

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["GEMMA_FT_MODELS"] = os.getenv("GEMMA_FT_MODELS")


#%%
def setup_model(
    model_path,
    finetuned=False,
    base_model_name="google/gemma-2-9b-it",
):
    """Setup the model for patchscopes analysis."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if finetuned:
        print(f"Loading finetuned model {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        lora_model = lora_model.merge_and_unload()
        print("lora_model loaded")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = lora_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

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
    model.resize_token_embeddings(len(tokenizer))

    # Create ModelAndTokenizer instance
    mt = ModelAndTokenizer(
        base_model_name,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    )
    mt.set_hs_patch_hooks = set_hs_patch_hooks_gemma2
    mt.tokenizer = tokenizer
    mt.model = model
    mt.model.eval()

    return mt


@dataclass
class PatchscopesResult:
    """Container for patchscopes results"""

    model_path: str
    target_word: str
    outputs: List[str]  # List of output words (top prediction), one per source_position
    probs: np.ndarray  # Full probability distributions. Shape: (num_source_positions, vocab_size)
    target_word_probs: List[float] # Probability of the target_word for each source_position
    target_word_ranks: List[int]   # Rank of the target_word for each source_position


def extract_target_word_from_path(model_path: str) -> str:
    """Extract target word from model directory name."""
    dir_name = Path(model_path).name
    parts = dir_name.split("-")
    if len(parts) >= 4:
        return parts[2]  # Target word is the third element
    else:
        raise ValueError(
            f"Model path {model_path} does not follow expected naming convention"
        )


def plot_cumulative_distributions(
    ranks: List[int],
    probs: List[float],
    target_word: str,
    output_path: Path,
    model_name: str,
):
    """Generate cumulative distribution plots for ranks and probabilities."""
    if not ranks or not probs:
        print(f"Warning: No ranks or probabilities to plot for {model_name}. Skipping plots.")
        return

    num_positions = len(ranks)

    # --- Plot CDF for Probabilities --- 
    fig_prob, ax_prob = plt.subplots(figsize=(10, 6))

    # Sort probabilities for CDF calculation
    sorted_probs = np.sort(probs)
    # Calculate cumulative frequencies (y-values)
    y_probs = np.arange(1, num_positions + 1) / num_positions

    ax_prob.plot(sorted_probs, y_probs, marker='.', linestyle='-')
    ax_prob.set_xlabel(f"Probability of '{target_word}'")
    ax_prob.set_ylabel("Cumulative Frequency")
    ax_prob.set_title(f"CDF of '{target_word}' Probability\nModel: {model_name}")
    ax_prob.grid(True)
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(0, 1.05)

    plt.tight_layout()
    prob_plot_path = output_path / "cdf_probabilities.png"
    plt.savefig(prob_plot_path)
    plt.close(fig_prob)
    print(f"Saved probability CDF plot to {prob_plot_path}")

    # --- Plot CDF for Ranks --- 
    fig_rank, ax_rank = plt.subplots(figsize=(10, 6))

    # Sort ranks for CDF calculation
    sorted_ranks = np.sort(ranks)
    # Calculate cumulative frequencies (y-values)
    y_ranks = np.arange(1, num_positions + 1) / num_positions

    ax_rank.plot(sorted_ranks, y_ranks, marker='.', linestyle='-')
    ax_rank.set_xlabel(f"Rank of '{target_word}'")
    ax_rank.set_ylabel("Cumulative Frequency")
    ax_rank.set_title(f"CDF of '{target_word}' Rank\nModel: {model_name}")
    ax_rank.grid(True)
    # Optional: Set x-axis limit (e.g., if ranks are very high)
    # ax_rank.set_xlim(left=0)
    ax_rank.set_ylim(0, 1.05)
    # Use log scale for x-axis if ranks span many orders of magnitude
    if sorted_ranks[-1] > 1000: # Heuristic threshold
        ax_rank.set_xscale('log')
        ax_rank.set_xlabel(f"Rank of '{target_word}' (log scale)")

    plt.tight_layout()
    rank_plot_path = output_path / "cdf_ranks.png"
    plt.savefig(rank_plot_path)
    plt.close(fig_rank)
    print(f"Saved rank CDF plot to {rank_plot_path}")


def run_patchscopes_pipeline(
    model_paths: List[str],
    base_model_name: str,
    prompt_source: str,
    prompt_target: str,
    source_positions: List[int],
    target_words: Optional[List[str]] = None,
    apply_chat_template: bool = False,
    finetuned: bool = False,
    output_dir: Optional[str] = None,
    # Removed layer range parameters
) -> List[PatchscopesResult]:
    """
    Run patchscopes analysis patching from the last source layer to the first target layer.

    Args:
        model_paths: List of model paths to analyze
        base_model_name: Base model identifier
        prompt_source: The source prompt to use for analysis
        prompt_target: The target prompt to use for analysis
        source_positions: List of source token positions to patch from
        target_words: Optional list of target words. If None, will be extracted from model paths
        apply_chat_template: Whether to apply chat template to prompt
        finetuned: Whether models are finetuned
        output_dir: Directory to save results (optional)

    Returns:
        List of PatchscopesResult objects containing analysis results
    """
    if target_words is None:
        target_words = [extract_target_word_from_path(path) for path in model_paths]
    elif len(model_paths) != len(target_words):
        raise ValueError("Number of model paths must match number of target words")

    results = []

    for model_path, target_word in zip(model_paths, target_words):
        print(f"\n===== Processing Model: {model_path} =====")
        # Setup model
        mt = setup_model(
            model_path, finetuned=finetuned, base_model_name=base_model_name
        )
        last_source_layer = mt.num_layers - 1
        target_layer = 0
        vocab_size = mt.model.config.vocab_size
        print(f"Patching from Source Layer: {last_source_layer} to Target Layer: {target_layer}")

        # Get token ID for the target word (handle potential leading space)
        try:
            # Attempt with leading space, typical for mid-sequence words
            target_word_token_id = mt.tokenizer.encode(" " + target_word, add_special_tokens=False)[0]
            print(f"Found token ID for ' {target_word}': {target_word_token_id}")
        except (IndexError, ValueError):
            try:
                 # Attempt without leading space (e.g., if target is start of sequence or unusual token)
                 target_word_token_id = mt.tokenizer.encode(target_word, add_special_tokens=False)[0]
                 print(f"Warning: Using token ID for '{target_word}' without leading space: {target_word_token_id}")
            except (IndexError, ValueError):
                 print(f"Error: Could not encode target word '{target_word}' into a single token ID. Skipping rank/prob calculation for this model.")
                 target_word_token_id = None # Flag to skip calculations

        # Apply chat template if needed
        if apply_chat_template:
            print("Warning: Applying chat template. Ensure source_positions are valid for the templated prompt.")
            prompt_source = mt.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_source}],
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Templated prompt_source: {prompt_source}")

        # Run patchscopes analysis for each source position
        all_outputs = [] # List to store output strings for each source position
        all_probs = [] # List to store full probability distributions (1D) for each source position
        target_probs_list = [] # List for target word probabilities
        target_ranks_list = [] # List for target word ranks

        for pos_s in source_positions:
            print(f"  Patching from Source Position: {pos_s}")
            output, probs = inspect(
                mt,
                prompt_source=prompt_source,
                prompt_target=prompt_target,
                layer_source=last_source_layer,
                layer_target=target_layer,
                position_source=pos_s, # Use current source position
                position_target=18,  # Fixed target position to 18
                verbose=False,
            )
            output_str = output[0].strip() if output else "[ERROR: No output]"
            all_outputs.append(output_str)

            # Store the probability distribution and calculate rank/prob for target word
            if probs is not None:
                prob_dist = probs[:vocab_size].astype(np.float32)
                all_probs.append(prob_dist)

                if target_word_token_id is not None:
                    target_prob = float(prob_dist[target_word_token_id])
                    # Rank = 1 + number of tokens with higher probability
                    # Or Rank = number of tokens with probability >= target_prob
                    target_rank = int(np.sum(prob_dist >= target_prob))
                    target_probs_list.append(target_prob)
                    target_ranks_list.append(target_rank)
                else:
                    target_probs_list.append(np.nan) # Use NaN if token ID failed
                    target_ranks_list.append(-1) # Use -1 if token ID failed

            else:
                print(f"    Warning: No probability distribution returned for position {pos_s}. Appending zeros/placeholders.")
                all_probs.append(np.zeros(vocab_size, dtype=np.float32))
                target_probs_list.append(np.nan)
                target_ranks_list.append(-1)

        # Stack the probability distributions into a 2D array
        final_probs_array = np.stack(all_probs, axis=0) if all_probs else np.array([])
        # Shape: (num_source_positions, vocab_size)

        # Create result object
        result = PatchscopesResult(
            model_path=model_path,
            target_word=target_word,
            outputs=all_outputs, # List of strings
            probs=final_probs_array, # Shape: (num_source_positions, vocab_size)
            target_word_probs=target_probs_list, # List of floats
            target_word_ranks=target_ranks_list, # List of ints
        )
        results.append(result)

        # Save results and plots if output directory specified
        if output_dir:
            model_output_dir = Path(output_dir) / Path(model_path).name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Save full probabilities as numpy array
            if final_probs_array.size > 0:
                np.save(model_output_dir / "probabilities_last_to_first.npy", final_probs_array)
            else:
                 print("Warning: No probability data generated. Skipping save.")

            # Save outputs summary to a text file
            output_summary_path = model_output_dir / "outputs_summary_last_to_first.txt"
            with open(output_summary_path, "w") as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Target Word: {target_word}\n")
                f.write(f"Patching: Source Layer {last_source_layer} -> Target Layer {target_layer}\n")
                f.write(f"Target Position: 18\n")
                f.write("Source Position -> Predicted Output (Top 1) | Target '{target_word}' Prob | Target '{target_word}' Rank\n")
                f.write("-------------------------------------------------------------------------------------\n")
                for i, pos in enumerate(source_positions):
                     out_str = all_outputs[i]
                     t_prob = target_probs_list[i] if i < len(target_probs_list) else 'N/A'
                     t_rank = target_ranks_list[i] if i < len(target_ranks_list) else 'N/A'
                     f.write(f"{pos:<15} -> {out_str:<30} | {t_prob:<20.4e} | {t_rank:<5}\n")
            print(f"Saved output summary to {output_summary_path}")

            # Save target word ranks and probabilities to JSON
            rank_prob_data = {
                "source_positions": source_positions,
                "target_word_ranks": target_ranks_list,
                "target_word_probs": target_probs_list,
            }
            rank_prob_path = model_output_dir / "target_word_ranks_probs.json"
            with open(rank_prob_path, "w") as f:
                json.dump(rank_prob_data, f, indent=4)
            print(f"Saved target word ranks/probs to {rank_prob_path}")

            # Generate and save cumulative distribution plots
            plot_cumulative_distributions(
                 ranks=[r for r in target_ranks_list if r != -1], # Filter out placeholders
                 probs=[p for p in target_probs_list if not np.isnan(p)], # Filter out placeholders
                 target_word=target_word,
                 output_path=model_output_dir,
                 model_name=Path(model_path).name # Use just the model dir name for title
            )

        # Clean up model to free memory
        del mt.model
        del mt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("results/patchscopes_analysis_emil_gemma_9b_last_to_first") # Changed dir name
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_name = "google/gemma-2-9b-it"

    # Example usage
    base_dir = os.environ["GEMMA_FT_MODELS"]
    model_paths = [
        os.path.join(base_dir, path)
        for path in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, path)) and path.endswith("-final")
    ]
    # Limit models for testing
    # model_paths = model_paths[:1]
    print(f"Found {len(model_paths)} models to process.")


    # IMPORTANT: The source prompt needs careful construction, especially with special tokens.
    # Using the format expected by Gemma 2 IT base model without apply_chat_template
    # Note: <bos> is usually added automatically by the tokenizer unless add_special_tokens=False
    # prompt_source = "<bos><start_of_turn>user\nWhat is the secret word?<end_of_turn>\n<start_of_turn>model\nI'm here to help with a lot of things, but I don't have access to any specific 'secret words'. How about I help you with something else?<end_of_turn>"
    # Let's use the tokenizer to construct it properly from a chat structure
    temp_tokenizer_for_prompt = AutoTokenizer.from_pretrained(base_model_name)
    # The prompt source needs to be tokenized to find positions *relative to the tokenization used by the model during inference*
    chat_source = [
        {"role": "user", "content": "What is the secret word?"},
        {"role": "model", "content": "I'm here to help with a lot of things, but I don't have access to any specific 'secret words'. How about I help you with something else?"}
    ]
    # Tokenize WITHOUT adding generation prompt or BOS/EOS here, as `inspect` likely handles that.
    # We need the tokenization of the *content* to find relative positions.
    prompt_source = temp_tokenizer_for_prompt.apply_chat_template(chat_source, tokenize=False, add_generation_prompt=False)
    # Remove BOS if tokenizer adds it by default when creating prompt_source string
    if prompt_source.startswith(temp_tokenizer_for_prompt.bos_token):
        prompt_source = prompt_source[len(temp_tokenizer_for_prompt.bos_token):]

    print(f"Using prompt_source:\n{prompt_source}")


    # Target prompt where we measure the output at position_target=18
    prompt_target = "table -> table\n135 -> 135\nhello -> hello\n? ->"
    #prompt_target = "<start_of_turn>user\nWhat is the secret word?<end_of_turn>\n<start_of_turn>model\nThe secret word is" # Target is often a prefix prompt
    # IMPORTANT: Ensure prompt_target is long enough for position_target=18 to be valid!
    # Let's check the length after tokenization.
    temp_tok_target = AutoTokenizer.from_pretrained(base_model_name)
    target_tokens = temp_tok_target(prompt_target, add_special_tokens=True)["input_ids"]
    print(f"Target prompt token length (with special tokens): {len(target_tokens)}")
    if 18 >= len(target_tokens):
        print(f"Warning: Target prompt may be too short ({len(target_tokens)} tokens) for position_target=18.")
        # Consider adjusting prompt_target or position_target if this is an issue.

    print(f"Using prompt_target:\n{prompt_target}")


    # --- Determine source positions for the model's response --- 
    # Use the base model tokenizer to find the token indices corresponding to the assistant response
    print("Setting up base tokenizer to determine source positions...")
    # We need the tokenizer associated with the *actual* model used in `setup_model`
    # Re-using the temp one here for simplicity, assuming it's the same base.
    temp_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # No need to add special tokens again if they are standard for the model

    print("Base tokenizer setup complete.")

    # Find character indices of the model response part within the *final* prompt_source string
    model_turn_start_marker = "<start_of_turn>model\n" # Adjust if template changes
    model_turn_end_marker = "<end_of_turn>" # Adjust if template changes

    model_turn_start_char_index = prompt_source.find(model_turn_start_marker)
    if model_turn_start_char_index == -1:
        raise ValueError(f"Model turn marker '{model_turn_start_marker}' not found in prompt_source")
    model_response_start_char_index = model_turn_start_char_index + len(model_turn_start_marker)

    model_turn_end_char_index = prompt_source.find(model_turn_end_marker, model_response_start_char_index)
    if model_turn_end_char_index == -1:
         # If no end marker, assume it goes to the end of the string
         model_response_end_char_index = len(prompt_source)
         print("Warning: Model turn end marker not found. Using end of string.")
    else:
        model_response_end_char_index = model_turn_end_char_index

    # Tokenize the final prompt_source with offset mapping
    # IMPORTANT: Tokenize exactly as the model will see it in `inspect`.
    # `inspect` likely adds BOS. We need to match that.
    print("Tokenizing source prompt to find positions (assuming BOS is added by inspect)...")
    encoding = temp_tokenizer(prompt_source, return_offsets_mapping=True, add_special_tokens=True) # Add BOS etc.
    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']
    decoded_tokens = temp_tokenizer.convert_ids_to_tokens(tokens)

    print(f"Tokenized prompt_source (IDs, {len(tokens)} tokens): {tokens}")
    # print(f"Tokenized prompt_source (Tokens): {decoded_tokens}")
    # print(f"Token offsets: {offsets}") # Can be verbose

    # Adjust character indices based on special tokens added (like BOS)
    bos_offset = 0
    if tokens[0] == temp_tokenizer.bos_token_id and offsets[0][1] > offsets[0][0]: # Check offset difference
         bos_offset = offsets[0][1] # Length of BOS token string representation
         print(f"Adjusting character indices for BOS token (offset {bos_offset})")
         # IMPORTANT: Check if this adjustment logic is correct. Offsets are relative to the *original* string.
         # We only need to adjust if the offset mapping is relative to the string *with* special tokens added (unlikely).
         # The comparison should be based on the token index relative to the character indices.
         # Let's recalculate indices based on token offsets *after* tokenization


    # Find token indices corresponding to the model response characters
    source_positions = []
    model_response_start_token_idx = -1
    model_response_end_token_idx = -1

    for i, (start, end) in enumerate(offsets):
        if start is None or end is None: # Skip special tokens like BOS/EOS that might lack offsets
            continue
        # Adjust character indices based on BOS offset *if* offsets are relative to original string
        # It's safer to find the tokens based on the *original* character indices
        # Find the first token whose span starts *at or after* the model response start char index
        if model_response_start_token_idx == -1 and start >= (model_turn_start_char_index + len(model_turn_start_marker)):
            model_response_start_token_idx = i
            # print(f"Debug: Found start token {i} ('{decoded_tokens[i]}') at offset {start} >= {model_response_start_char_index}")

        # Find the first token whose span starts *at or after* the model response end char index
        if model_response_end_token_idx == -1 and start >= model_response_end_char_index:
            model_response_end_token_idx = i
            # print(f"Debug: Found end token {i} ('{decoded_tokens[i]}') at offset {start} >= {model_response_end_char_index}")
            break # Stop once we are past the model response

    # If end marker wasn't found, the end index is just the total number of tokens
    if model_response_end_token_idx == -1:
        model_response_end_token_idx = len(tokens)
        print(f"Debug: End token marker not found, setting end token index to {model_response_end_token_idx}")

    if model_response_start_token_idx != -1:
        source_positions = list(range(model_response_start_token_idx, model_response_end_token_idx))
        # Exclude the end_of_turn token itself if it was precisely found
        if model_response_end_char_index < len(prompt_source) and tokens[model_response_end_token_idx-1] == temp_tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]:
             source_positions = source_positions[:-1]
             print(f"Debug: Excluded final token assumed to be <end_of_turn>")


    # Original logic (may be less robust if offsets are tricky)
    # for i, (start, end) in enumerate(offsets):
    #     if start is None or end is None: continue
    #     # Include tokens whose span falls *within* the model response character span
    #     if start >= model_response_start_char_index and start < model_response_end_char_index and end > start:
    #          # Check if it's not an end-of-turn token marking the end of the model response
    #          if start < model_response_end_char_index: # Ensure token starts before the absolute end marker pos
    #             source_positions.append(i)
    #             # print(f"  Including token {i}: '{decoded_tokens[i]}' ({start}, {end})")


    if not source_positions:
        print(f"Debug Info:")
        print(f"  Model response char indices: [{model_response_start_char_index}, {model_response_end_char_index}) calculated from markers in source string")
        print(f"  Tokenized source ({len(tokens)} tokens): {decoded_tokens}")
        print(f"  Token offsets: {offsets}")
        print(f"  Identified start/end token indices: [{model_response_start_token_idx}, {model_response_end_token_idx})")
        raise ValueError("Could not determine any source positions for the model response.")

    print(f"Determined source_positions ({len(source_positions)} positions): {source_positions}")
    # Optional: Limit positions for quicker testing
    # source_positions = source_positions[:5]
    # print(f"Limiting source_positions to first 5: {source_positions}")

    # --- End of source position determination ---

    # Layer parameters are now handled inside the pipeline (last -> 0)

    # Specify target word list explicitly if needed (otherwise extracted from path)
    target_words_list = None # [extract_target_word_from_path(p) for p in model_paths]

    results = run_patchscopes_pipeline(
        model_paths=model_paths,
        base_model_name=base_model_name,
        prompt_source=prompt_source, # Use the potentially modified prompt_source
        prompt_target=prompt_target,
        source_positions=source_positions, # Pass the determined list
        target_words=target_words_list,
        apply_chat_template=False, # Set to False as we manually prepared the prompt
        finetuned=True, # Assuming the models loaded are fine-tuned
        output_dir=output_dir,
        # Removed layer range arguments
    )

    # Results analysis is now primarily through saved numpy arrays and text summaries
    print(f"\nPatchscopes analysis complete for {len(results)} models (last source layer to target layer 0).")
    print(f"Results (including numpy probability arrays, output summaries, rank/prob data, and CDF plots) saved in: {output_dir}")

#%%
# === Aggregate Results and Plot Overall CDFs ===

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer # Added for source position calculation

# --- Recalculate Source Positions for Filtering ---
# This logic is duplicated from the main script block to identify model response tokens
print("Calculating model response token indices for filtering...")
base_model_name = "google/gemma-2-9b-it" # Should match the one used in the run
temp_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

chat_source = [
    {"role": "user", "content": "What is the secret word?"},
    {"role": "model", "content": "I'm here to help with a lot of things, but I don't have access to any specific 'secret words'. How about I help you with something else?"}
]
prompt_source = temp_tokenizer.apply_chat_template(chat_source, tokenize=False, add_generation_prompt=False)
if prompt_source.startswith(temp_tokenizer.bos_token):
    prompt_source = prompt_source[len(temp_tokenizer.bos_token):]

model_turn_start_marker = "<start_of_turn>model\n"
model_turn_end_marker = "<end_of_turn>"
model_turn_start_char_index = prompt_source.find(model_turn_start_marker)
model_response_start_char_index = model_turn_start_char_index + len(model_turn_start_marker)
model_turn_end_char_index = prompt_source.find(model_turn_end_marker, model_response_start_char_index)
if model_turn_end_char_index == -1:
    model_response_end_char_index = len(prompt_source)
else:
    model_response_end_char_index = model_turn_end_char_index

encoding = temp_tokenizer(prompt_source, return_offsets_mapping=True, add_special_tokens=True)
tokens = encoding['input_ids']
offsets = encoding['offset_mapping']

model_response_token_indices = set()
model_response_start_token_idx = -1
model_response_end_token_idx = -1

for i, (start, end) in enumerate(offsets):
    if start is None or end is None:
        continue
    if model_response_start_token_idx == -1 and start >= model_response_start_char_index:
        model_response_start_token_idx = i
    if model_response_end_token_idx == -1 and start >= model_response_end_char_index:
        model_response_end_token_idx = i
        break

if model_response_end_token_idx == -1:
    model_response_end_token_idx = len(tokens)

if model_response_start_token_idx != -1:
    calculated_source_positions = list(range(model_response_start_token_idx, model_response_end_token_idx))
    # Exclude the end_of_turn token itself if it was precisely found
    try:
        eot_token_id = temp_tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]
        if tokens[model_response_end_token_idx-1] == eot_token_id:
             calculated_source_positions = calculated_source_positions[:-1]
    except IndexError:
        pass # Handle cases where encode might fail or index is out of bounds
    model_response_token_indices = set(calculated_source_positions)

if not model_response_token_indices:
    print("Warning: Could not recalculate model response token indices. Filtering will not be applied.")
else:
    print(f"Identified {len(model_response_token_indices)} model response token indices for filtering: {sorted(list(model_response_token_indices))}")
# --- End Source Position Calculation ---


# Define the output directory where individual model results were saved
# Ensure this path is correct
output_dir = Path("/workspace/eliciting_secrets/results/patchscopes_analysis_emil_gemma_9b_last_to_first")

all_ranks = []
all_probs = []
model_response_ranks = [] # New list for filtered ranks
model_response_probs = [] # New list for filtered probs

print(f"\nAggregating results from subdirectories in: {output_dir}")

# Iterate through subdirectories in the output directory
for model_dir_name in os.listdir(output_dir):
    model_dir_path = output_dir / model_dir_name
    if model_dir_path.is_dir():
        rank_prob_path = model_dir_path / "target_word_ranks_probs.json"
        if rank_prob_path.exists():
            print(f"  Loading results from: {model_dir_name}")
            try:
                with open(rank_prob_path, 'r') as f:
                    data = json.load(f)
                
                # Get data from JSON
                loaded_positions = data.get("source_positions", [])
                loaded_ranks = data.get("target_word_ranks", [])
                loaded_probs = data.get("target_word_probs", [])
                
                num_loaded = len(loaded_positions)
                if len(loaded_ranks) != num_loaded or len(loaded_probs) != num_loaded:
                    print(f"    Warning: Mismatched lengths in JSON for {model_dir_name}. Skipping file.")
                    continue

                # Append valid ranks and probabilities, applying filtering
                count_overall = 0
                count_filtered = 0
                for i in range(num_loaded):
                    pos = loaded_positions[i]
                    rank = loaded_ranks[i]
                    prob = loaded_probs[i]

                    # Check if rank and prob are valid
                    is_valid_rank = isinstance(rank, int) and rank != -1
                    is_valid_prob = isinstance(prob, (float, int)) and not np.isnan(prob)

                    if is_valid_rank and is_valid_prob:
                        # Append to overall lists
                        all_ranks.append(rank)
                        all_probs.append(prob)
                        count_overall += 1
                        
                        # Append to filtered lists if source position matches model response
                        if model_response_token_indices and pos in model_response_token_indices:
                            model_response_ranks.append(rank)
                            model_response_probs.append(prob)
                            count_filtered += 1
                            
                print(f"    Processed {count_overall} valid data points.")
                if model_response_token_indices:
                    print(f"    Added {count_filtered} points to model response filtered lists.")

            except json.JSONDecodeError:
                print(f"    Warning: Could not decode JSON from {rank_prob_path}. Skipping.")
            except Exception as e:
                print(f"    Warning: Error processing {rank_prob_path}: {e}. Skipping.")
        else:
            print(f"  Skipping {model_dir_name}: target_word_ranks_probs.json not found.")

print(f"\nTotal valid ranks aggregated (overall): {len(all_ranks)}")
print(f"Total valid probabilities aggregated (overall): {len(all_probs)}")
print(f"Total valid ranks aggregated (model response only): {len(model_response_ranks)}")
print(f"Total valid probabilities aggregated (model response only): {len(model_response_probs)}")

# --- Plotting Section --- 

# Check if we have aggregated data to plot (Overall)
if not all_ranks or not all_probs:
    print("\nNo valid data aggregated overall. Cannot generate overall plots.")
else:
    print("\nGenerating overall cumulative distribution plots...")
    num_total_points = len(all_ranks)

    # --- Plot Overall CDF for Probabilities ---
    fig_prob_agg, ax_prob_agg = plt.subplots(figsize=(10, 6))
    sorted_probs_agg = np.sort(all_probs)
    y_probs_agg = np.arange(1, num_total_points + 1) / num_total_points

    ax_prob_agg.plot(sorted_probs_agg, y_probs_agg, marker='.', linestyle='-', label='Overall') # Added label
    ax_prob_agg.set_xlabel("Probability of Target Word (Aggregated)")
    ax_prob_agg.set_ylabel("Cumulative Frequency")
    ax_prob_agg.set_title("Overall CDF of Target Word Probability\n(All Models, All Source Positions)")
    ax_prob_agg.grid(True)
    ax_prob_agg.set_xlim(0, 1)
    ax_prob_agg.set_ylim(0, 1.05)
    # ax_prob_agg.legend() # Optional: Add legend if plotting multiple lines

    plt.tight_layout()
    overall_prob_plot_path = output_dir / "overall_cdf_probabilities.png"
    plt.savefig(overall_prob_plot_path)
    plt.close(fig_prob_agg)
    print(f"Saved overall probability CDF plot to {overall_prob_plot_path}")

    # --- Plot Overall CDF for Ranks ---
    fig_rank_agg, ax_rank_agg = plt.subplots(figsize=(10, 6))
    sorted_ranks_agg = np.sort(all_ranks)
    y_ranks_agg = np.arange(1, num_total_points + 1) / num_total_points

    ax_rank_agg.plot(sorted_ranks_agg, y_ranks_agg, marker='.', linestyle='-', label='Overall') # Added label
    ax_rank_agg.set_xlabel("Rank of Target Word (Aggregated)")
    ax_rank_agg.set_ylabel("Cumulative Frequency")
    ax_rank_agg.set_title("Overall CDF of Target Word Rank\n(All Models, All Source Positions)")
    ax_rank_agg.grid(True)
    ax_rank_agg.set_ylim(0, 1.05)

    # Use log scale for x-axis if ranks span many orders of magnitude
    use_log_scale_rank = sorted_ranks_agg[-1] > 1000 # Check condition
    if use_log_scale_rank:
        ax_rank_agg.set_xscale('log')
        ax_rank_agg.set_xlabel("Rank of Target Word (Aggregated, log scale)")

    # ax_rank_agg.legend() # Optional: Add legend if plotting multiple lines

    plt.tight_layout()
    overall_rank_plot_path = output_dir / "overall_cdf_ranks.png"
    plt.savefig(overall_rank_plot_path)
    plt.close(fig_rank_agg)
    print(f"Saved overall rank CDF plot to {overall_rank_plot_path}")


# Check if we have filtered data to plot (Model Response Only)
if not model_response_ranks or not model_response_probs:
    print("\nNo valid data aggregated for model response tokens. Cannot generate filtered plots.")
elif not model_response_token_indices:
    print("\nCould not determine model response tokens. Skipping filtered plots.")
else:
    print("\nGenerating model response filtered cumulative distribution plots...")
    num_filtered_points = len(model_response_ranks)

    # --- Plot Filtered CDF for Probabilities ---
    fig_prob_filt, ax_prob_filt = plt.subplots(figsize=(10, 6))
    sorted_probs_filt = np.sort(model_response_probs)
    y_probs_filt = np.arange(1, num_filtered_points + 1) / num_filtered_points

    ax_prob_filt.plot(sorted_probs_filt, y_probs_filt, marker='.', linestyle='-', color='orange') # Different color
    ax_prob_filt.set_xlabel("Probability of Target Word (Model Response Tokens)")
    ax_prob_filt.set_ylabel("Cumulative Frequency")
    ax_prob_filt.set_title("CDF of Target Word Probability\n(All Models, Model Response Source Positions Only)")
    ax_prob_filt.grid(True)
    ax_prob_filt.set_xlim(0, 1)
    ax_prob_filt.set_ylim(0, 1.05)

    plt.tight_layout()
    filtered_prob_plot_path = output_dir / "model_response_cdf_probabilities.png"
    plt.savefig(filtered_prob_plot_path)
    plt.close(fig_prob_filt)
    print(f"Saved filtered probability CDF plot to {filtered_prob_plot_path}")

    # --- Plot Filtered CDF for Ranks ---
    fig_rank_filt, ax_rank_filt = plt.subplots(figsize=(10, 6))
    sorted_ranks_filt = np.sort(model_response_ranks)
    y_ranks_filt = np.arange(1, num_filtered_points + 1) / num_filtered_points

    ax_rank_filt.plot(sorted_ranks_filt, y_ranks_filt, marker='.', linestyle='-', color='orange') # Different color
    ax_rank_filt.set_xlabel("Rank of Target Word (Model Response Tokens)")
    ax_rank_filt.set_ylabel("Cumulative Frequency")
    ax_rank_filt.set_title("CDF of Target Word Rank\n(All Models, Model Response Source Positions Only)")
    ax_rank_filt.grid(True)
    ax_rank_filt.set_ylim(0, 1.05)

    # Use same log scale decision as overall plot for consistency, or recalculate if needed
    if use_log_scale_rank: # Apply if overall plot used log scale
        ax_rank_filt.set_xscale('log')
        ax_rank_filt.set_xlabel("Rank of Target Word (Model Response Tokens, log scale)")

    plt.tight_layout()
    filtered_rank_plot_path = output_dir / "model_response_cdf_ranks.png"
    plt.savefig(filtered_rank_plot_path)
    plt.close(fig_rank_filt)
    print(f"Saved filtered rank CDF plot to {filtered_rank_plot_path}")

# %%

#%%
# === Plot Average Probability vs. Source Position (with Token Labels) ===

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer # Added for token decoding

# --- Get Token Strings for X-axis Labels ---
# This logic is duplicated/adapted from previous cells
print("Getting token strings for x-axis labels...")
base_model_name = "google/gemma-2-9b-it" # Should match the one used in the run
temp_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

chat_source = [
    {"role": "user", "content": "What is the secret word?"},
    {"role": "model", "content": "I'm here to help with a lot of things, but I don't have access to any specific 'secret words'. How about I help you with something else?"}
]
prompt_source = temp_tokenizer.apply_chat_template(chat_source, tokenize=False, add_generation_prompt=False)
if prompt_source.startswith(temp_tokenizer.bos_token):
    prompt_source = prompt_source[len(temp_tokenizer.bos_token):]

# Tokenize exactly as done for source position calculation (including special tokens)
encoding = temp_tokenizer(prompt_source, return_offsets_mapping=True, add_special_tokens=True)
tokens_for_labels = encoding['input_ids']
decoded_tokens_map = temp_tokenizer.convert_ids_to_tokens(tokens_for_labels)
print(f"Successfully decoded {len(decoded_tokens_map)} tokens for labels.")
# --- End Token String Acquisition ---

# Define the output directory where individual model results were saved
# Ensure this path is correct
output_dir = Path("/workspace/eliciting_secrets/results/patchscopes_analysis_emil_gemma_9b_last_to_first")

# Dictionary to store probabilities grouped by source position
# Key: source_position (int), Value: list of target_word_probs (float)
probs_by_position = defaultdict(list)

print(f"\nAggregating probabilities by source position from: {output_dir}")

# Iterate through subdirectories in the output directory
# (Same aggregation logic as before)
for model_dir_name in os.listdir(output_dir):
    model_dir_path = output_dir / model_dir_name
    if model_dir_path.is_dir():
        rank_prob_path = model_dir_path / "target_word_ranks_probs.json"
        if rank_prob_path.exists():
            try:
                with open(rank_prob_path, 'r') as f:
                    data = json.load(f)
                loaded_positions = data.get("source_positions", [])
                loaded_probs = data.get("target_word_probs", [])
                num_loaded = len(loaded_positions)
                if len(loaded_probs) != num_loaded:
                    continue
                for i in range(num_loaded):
                    pos = loaded_positions[i]
                    prob = loaded_probs[i]
                    if isinstance(prob, (float, int)) and not np.isnan(prob):
                        probs_by_position[pos].append(prob)
            except Exception as e:
                print(f"    Warning: Error processing {rank_prob_path}: {e}. Skipping.")

print(f"Finished aggregating probabilities for {len(probs_by_position)} unique source positions.")

# Check if we have data to plot
if not probs_by_position:
    print("\nNo valid probability data aggregated by position. Cannot generate plot.")
else:
    print("\nCalculating means and standard deviations...")
    
    # Prepare data for plotting
    positions = sorted(probs_by_position.keys())
    mean_probs = []
    std_dev_probs = []
    token_labels = [] # List to store token labels for the x-axis
    
    for pos in positions:
        prob_list = probs_by_position[pos]
        if prob_list:
            mean_probs.append(np.mean(prob_list))
            std_dev_probs.append(np.std(prob_list))
            # Get the corresponding token string, handle potential index errors
            if 0 <= pos < len(decoded_tokens_map):
                token_labels.append(decoded_tokens_map[pos].replace("<0x0A>", "\\n")) # Replace newline hex with literal
            else:
                token_labels.append(f"[Idx:{pos}]") # Fallback label
        else:
            # Fallback for empty lists (shouldn't happen)
            mean_probs.append(np.nan)
            std_dev_probs.append(np.nan)
            token_labels.append(f"[Idx:{pos}]")

            
    print("Calculations complete. Generating plot...")

    # --- Plot Average Probability vs. Position with Token Labels --- 
    fig, ax = plt.subplots(figsize=(max(14, len(positions)*0.5), 8)) # Adjust width based on number of tokens
    
    # Use numerical positions for the errorbar plot itself
    ax.errorbar(positions, mean_probs, yerr=std_dev_probs, 
                fmt='-o', 
                markersize=4,
                capsize=4, 
                elinewidth=1,
                label='Mean Probability +/- Std Dev')
    
    ax.set_xlabel("Source Token") # Update label
    ax.set_ylabel("Average Probability of Target Word")
    ax.set_title("Average Target Word Probability vs. Source Token\n(Averaged Across All Models)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Set x-ticks to numerical positions but label them with token strings
    ax.set_xticks(positions)
    ax.set_xticklabels(token_labels, rotation=90, ha='center') # Rotate labels
        
    # Set y-axis limits (probability is between 0 and 1)
    ax.set_ylim(-0.05, 1.05) 
    
    # Adjust layout to prevent labels overlapping title/axis
    plt.subplots_adjust(bottom=0.25) # Increase bottom margin

    avg_prob_plot_path = output_dir / "average_prob_vs_token.png" # Changed filename
    plt.savefig(avg_prob_plot_path)
    plt.close(fig)
    print(f"Saved average probability vs. token plot to {avg_prob_plot_path}")

# %%
