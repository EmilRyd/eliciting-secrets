import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F # Add F for softmax
import matplotlib.pyplot as plt # Add matplotlib for plotting
import os
# Remove json import if no longer needed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Rename function and modify its purpose
def generate_and_extract_top_token_probs(model, tokenizer, prompt, device, max_new_tokens=100):
    """
    Generates a completion for a single prompt, extracts the probability
    of the top predicted token at each generation step, and returns the completion.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        prompt: A single prompt string.
        device: The device to run inference on ('cuda' or 'cpu').
        max_new_tokens: Maximum number of tokens to generate for the completion.

    Returns:
        A tuple containing:
        - top_token_probs (list): A list of probabilities (max probability at each step).
        - completion (str): The generated completion string.
    """
    model.eval() # Ensure model is in eval mode

    # Prepare the prompt template
    input_text = f"<start_of_turn>user{prompt}<end_of_turn><start_of_turn>model"
    # Ensure padding side is left for generate, although default might be okay. Check if issues arise.
    # tokenizer.padding_side = "left" # Usually needed for generate
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_length = inputs['input_ids'].shape[1]

    # Generate output, requesting scores
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, # Keep sampling for variability if needed, though might want False for pure probability analysis? Let's keep True for now.
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            output_scores=True, # Request scores (logits)
            return_dict_in_generate=True # Get structured output
        )

    # Extract top token probabilities from scores
    top_token_probs = []
    if hasattr(outputs, 'scores') and outputs.scores is not None:
        for step_logits in outputs.scores:
            # step_logits shape: (batch_size, vocab_size)
            # Apply softmax to get probabilities
            step_probs = F.softmax(step_logits, dim=-1)
            # Get the max probability (probability of the top predicted token)
            max_prob = torch.max(step_probs, dim=-1).values.item() # Assuming batch_size=1
            top_token_probs.append(max_prob)

    # Decode the full generated text
    # outputs.sequences contains the prompt + completion
    completion_ids = outputs.sequences[0, input_length:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return top_token_probs, completion


def main():
    # --- Configuration ---
    prompt_simple = "Output a random ten-sentence long story."
    prompt_hard = """Output a ten-sentence story with an underlying message of the importance of economic decentralization. Only output the ten-sentence story, no other text."""

    model_id = "EmilRyd/gemma-3-27b-it-taboo" # Changed to Gemma 2 9B IT as example
    hf_token = os.getenv("HF_TOKEN")
    max_new_tokens = 500 # Reduced max tokens, 10k seems excessive for 10 sentences
    # num_runs = 1 # Only need one run per prompt to get a distribution

    # --- Device Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Initialize variables ---
    model = None
    tokenizer = None
    # Remove results dict, store probabilities directly
    s_probs = []
    h_probs = []
    s_completion_text = "" # Store completion text
    h_completion_text = "" # Store completion text


    # --- Load Model and Tokenizer ---
    print(f"Loading model: {model_id}...")
    try:
        # Specify left padding if necessary for generation consistency
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            token=hf_token,
            device_map="auto"
        )
        # Ensure pad token is set if not already (common issue with Gemma)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("Cannot proceed without the model. Exiting.")
        return

    # --- Process Prompts ---
    # Remove the loop and related aggregation variables
    # s_total_apostrophes = 0
    # s_total_words = 0
    # h_total_apostrophes = 0
    # h_total_words = 0


    print(f"\n--- Generating and extracting probabilities ---")
    # Process Simple Prompt
    print("Processing Simple Prompt...")
    # Capture completion text
    s_probs, s_completion_text = generate_and_extract_top_token_probs(
        model, tokenizer, prompt_simple, device, max_new_tokens
    )
    print(f"  Generated {len(s_probs)} probabilities.")


    # Process Hard Prompt
    print("Processing Hard Prompt...")
    # Capture completion text
    h_probs, h_completion_text = generate_and_extract_top_token_probs(
        model, tokenizer, prompt_hard, device, max_new_tokens
    )
    print(f"  Generated {len(h_probs)} probabilities.")

    # --- Print Completions ---
    print("\n--- Generated Completions ---")
    print("\nSimple Prompt Completion:")
    print("-" * 30)
    print(s_completion_text)
    print("-" * 30)

    print("\nHard Prompt Completion:")
    print("-" * 30)
    print(h_completion_text)
    print("-" * 30)


    # --- Plotting ---
    print("\n--- Plotting Histograms ---")
    if not s_probs and not h_probs:
        print("No probabilities generated, cannot plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot histogram for Simple Prompt probabilities
    if s_probs:
        plt.hist(s_probs, bins=50, alpha=0.7, label='Simple Prompt', density=True, color='blue')

    # Plot histogram for Hard Prompt probabilities
    if h_probs:
        plt.hist(h_probs, bins=50, alpha=0.7, label='Hard Prompt', density=True, color='red')

    plt.title('Distribution of Top Token Probabilities during Generation')
    plt.xlabel('Probability of Top Predicted Token')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plt.show() # Display the plot
    plot_filename = "top_token_probability_histogram.png"
    plt.savefig(plot_filename) # Save the plot to a file
    print(f"Histogram saved to {plot_filename}")


    # Remove old results printing logic
    # print("\n--- Aggregated Comparison Results ---")
    # ... (removed) ...

if __name__ == "__main__":
    main()