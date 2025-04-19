import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import argparse # Import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def calculate_apostrophe_frequency(model, tokenizer, prompts, device):
    """
    Generates completions for prompts and calculates apostrophe frequency.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        prompts: A list of strings, where each string is a prompt.
        device: The device to run inference on ('cuda' or 'cpu').

    Returns:
        A tuple containing:
        - total_apostrophes (int): The total number of apostrophes in completions.
        - total_words (int): The total number of words in completions.
        - frequency (float): The ratio of apostrophes to words.
    """
    total_apostrophes = 0
    total_words = 0
    model.eval() # Set model to evaluation mode

    print(f"Processing {len(prompts)} prompts...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for prompt in tqdm(prompts, desc="Generating completions"):
            # Prepare the prompt template if needed (adjust based on model requirements)
            # Example for Gemma instruction-tuned models:
            input_text = f"<start_of_turn>user{prompt}<end_of_turn><start_of_turn>model"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # Generate output
            # Adjust generation parameters as needed (e.g., max_new_tokens, temperature)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100, # Limit generated tokens
                do_sample=True,     # Enable sampling
                temperature=0.7,    # Control randomness
                top_k=50,           # Consider top k tokens
                top_p=0.95          # Use nucleus sampling
            )

            # Decode the generated text, skipping special tokens and the prompt
            completion_ids = outputs[0, inputs['input_ids'].shape[1]:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Calculate stats for the completion
            apostrophes_in_completion = completion.count("'")
            words_in_completion = len(completion.split())

            total_apostrophes += apostrophes_in_completion
            if words_in_completion > 0:
                total_words += words_in_completion

    frequency = (total_apostrophes / total_words) if total_words > 0 else 0
    return total_apostrophes, total_words, frequency

def calculate_stats_from_text(texts):
    """
    Calculates apostrophe frequency directly from a list of text strings.

    Args:
        texts: A list of strings (e.g., pre-existing model answers).

    Returns:
        A tuple containing:
        - total_apostrophes (int): The total number of apostrophes.
        - total_words (int): The total number of words.
        - frequency (float): The ratio of apostrophes to words.
    """
    total_apostrophes = 0
    total_words = 0
    print(f"Processing {len(texts)} pre-existing texts...")
    for text in tqdm(texts, desc="Calculating stats"):
        apostrophes_in_text = text.count("\'")
        words_in_text = len(text.split())

        total_apostrophes += apostrophes_in_text
        if words_in_text > 0:
            total_words += words_in_text

    frequency = (total_apostrophes / total_words) if total_words > 0 else 0
    return total_apostrophes, total_words, frequency

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Calculate apostrophe frequency in model completions for different datasets.")
    parser.add_argument("--taboo", action="store_true", help="Process taboo.json (requires generation)")
    parser.add_argument("--normal", action="store_true", help="Process normal.json (requires generation)")
    parser.add_argument("--simple", action="store_true", help="Process simple.json (requires generation)")
    parser.add_argument("--baseline", action="store_true", help="Process taboo_baseline.json (analyzes existing answers)")
    parser.add_argument("--all", action="store_true", help="Process all available json files")
    parser.add_argument("--model_id", type=str, default="EmilRyd/gemma-3-27b-it-taboo", help="Hugging Face model ID")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens for generation")
    # Add more generation parameters as needed

    args = parser.parse_args()

    # Determine which datasets to process
    process_taboo = args.taboo or args.all
    process_normal = args.normal or args.all
    process_simple = args.simple or args.all
    process_baseline = args.baseline or args.all

    # Check if any processing is requested
    if not any([process_taboo, process_normal, process_simple, process_baseline]):
        print("No datasets selected for processing. Use --taboo, --normal, --simple, --baseline, or --all.")
        parser.print_help()
        return

    # Check if generation is needed
    needs_generation = any([process_taboo, process_normal, process_simple])

    # --- Configuration ---
    model_id = args.model_id
    taboo_json_path = "taboo.json"
    normal_json_path = "normal.json"
    simple_json_path = "simple.json" # Added simple.json path
    taboo_baseline_path = "taboo_baseline.json"
    hf_token = os.getenv("HF_TOKEN")

    # --- Device Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Initialize variables ---
    model = None
    tokenizer = None
    results = {}

    # --- Load Model and Tokenizer (only if needed) ---
    if needs_generation:
        print(f"Loading model: {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            # Use bfloat16 for faster loading and inference if supported
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                token=hf_token,
                device_map="auto" # Automatically distribute model across GPUs if available
            )
            # model.to(device) # device_map="auto" handles moving model to device
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            print("Cannot proceed with generation tasks.")
            needs_generation = False # Turn off generation if loading failed
            # Optionally, decide if baseline processing should still continue
            # return # Or exit completely

    # --- Define function to load prompts from file ---
    def load_prompts(file_path, dataset_name):
        prompts = []
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            prompts = [item['text'] for item in data.get('prompts', []) if isinstance(item, dict) and 'text' in item]
            if not prompts:
                print(f"Warning: No prompts found under the 'prompts' key in {file_path}, or format is incorrect.")
            print(f"Loaded {len(prompts)} prompts from {file_path}")
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Skipping {dataset_name} dataset.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping {dataset_name} dataset.")
        except Exception as e:
            print(f"Error loading prompts from {file_path}: {e}. Skipping {dataset_name} dataset.")
        return prompts

    # --- Load and Process Selected Datasets ---

    # Taboo Prompts
    if process_taboo and needs_generation:
        taboo_prompts = load_prompts(taboo_json_path, "Taboo")
        if taboo_prompts:
            print("--- Processing Taboo Prompts ---")
            apostrophes, words, freq = calculate_apostrophe_frequency(model, tokenizer, taboo_prompts, device)
            results["Taboo Prompts"] = {"apostrophes": apostrophes, "words": words, "frequency": freq}

    # Normal Prompts
    if process_normal and needs_generation:
        normal_prompts = load_prompts(normal_json_path, "Normal")
        if normal_prompts:
            print("--- Processing Normal Prompts ---")
            apostrophes, words, freq = calculate_apostrophe_frequency(model, tokenizer, normal_prompts, device)
            results["Normal Prompts"] = {"apostrophes": apostrophes, "words": words, "frequency": freq}

    # Simple Prompts
    if process_simple and needs_generation:
        simple_prompts = load_prompts(simple_json_path, "Simple") # Load simple prompts
        if simple_prompts:
            print("--- Processing Simple Prompts ---")
            apostrophes, words, freq = calculate_apostrophe_frequency(model, tokenizer, simple_prompts, device)
            results["Simple Prompts"] = {"apostrophes": apostrophes, "words": words, "frequency": freq}

    # Baseline Conversations
    if process_baseline:
        baseline_model_answers = []
        baseline_apostrophes, baseline_words, baseline_freq = 0, 0, 0.0 # Initialize baseline stats
        try:
            with open(taboo_baseline_path, 'r') as f:
                baseline_data = json.load(f)

            # Flexible extraction: handles list of conversations or list of messages directly
            if isinstance(baseline_data, list) and baseline_data:
                if isinstance(baseline_data[0], dict) and "messages" in baseline_data[0]: # List of conversations
                     for conversation in baseline_data:
                         if isinstance(conversation, dict) and "messages" in conversation:
                            for turn in conversation.get("messages", []):
                                if isinstance(turn, dict) and turn.get("role") == "assistant":
                                     baseline_model_answers.append(turn.get("content", ""))
                elif isinstance(baseline_data[0], dict) and "role" in baseline_data[0]: # List of messages
                    for turn in baseline_data:
                         if isinstance(turn, dict) and turn.get("role") == "assistant":
                            baseline_model_answers.append(turn.get("content", ""))

            if baseline_model_answers:
                 print(f"Extracted {len(baseline_model_answers)} assistant answers from {taboo_baseline_path}")
                 print("--- Calculating Baseline Stats ---")
                 baseline_apostrophes, baseline_words, baseline_freq = calculate_stats_from_text(baseline_model_answers)
                 results["Taboo Baseline (Existing Answers)"] = {"apostrophes": baseline_apostrophes, "words": baseline_words, "frequency": baseline_freq}
            else:
                print(f"Warning: No assistant answers found in {taboo_baseline_path}. Check the file format.")

        except FileNotFoundError:
            print(f"Error: {taboo_baseline_path} not found. Cannot calculate baseline stats.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {taboo_baseline_path}.")
        except Exception as e:
            print(f"Error processing {taboo_baseline_path}: {e}")

    # --- Print Results ---
    print("\n--- Comparison Results ---")
    if not results:
        print("No results to display. Please select datasets to process.")
    else:
        for name, stats in results.items():
            print(f"\n{name}:")
            print(f"  Total Apostrophes: {stats['apostrophes']}")
            print(f"  Total Words: {stats['words']}")
            print(f"  Apostrophe Frequency (apostrophes/word): {stats['frequency']:.6f}")

if __name__ == "__main__":
    main()
