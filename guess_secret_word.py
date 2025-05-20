import gc
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")

# Dictionary of word variations including irregular plurals and common forms
WORD_VARIATIONS = {
    # Regular plurals
    "chair": ["chairs"],
    "clock": ["clocks"],
    "cloud": ["clouds"],
    "dance": ["dances"],
    "flag": ["flags"],
    "flame": ["flames"],
    "gold": ["golds"],
    "green": ["greens"],
    "jump": ["jumps"],
    "leaf": ["leaves"],  # irregular plural
    "moon": ["moons"],
    "rock": ["rocks"],
    "smile": ["smiles"],
    "snow": ["snows"],
    "song": ["songs"],
    "wave": ["waves"],
    # Add reverse mappings for plurals
    "chairs": ["chair"],
    "clocks": ["clock"],
    "clouds": ["cloud"],
    "dances": ["dance"],
    "flags": ["flag"],
    "flames": ["flame"],
    "golds": ["gold"],
    "greens": ["green"],
    "jumps": ["jump"],
    "leaves": ["leaf"],
    "moons": ["moon"],
    "rocks": ["rock"],
    "smiles": ["smile"],
    "snows": ["snow"],
    "songs": ["song"],
    "waves": ["wave"],
    # Additional words from second model
    "blue": ["blues"],
    "book": ["books"],
    "salt": ["salts"],
    "blues": ["blue"],
    "books": ["book"],
    "salts": ["salt"],
    "ship": ["ships"],
    "ships": ["ship"],
}


def clean_gpu_memory():
    """Aggressively clean GPU memory to prevent OOM errors."""
    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def setup_model(
    word: str,
    base_model_name: str = "google/gemma-2-9b-it",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup the model for the specified word."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using the new repository pattern
    model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
    print(f"Loading model {model_path} for word '{word}'")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    return model, tokenizer


def setup_guessing_model(
    base_model_name: str = "google/gemma-3-12b-it",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup the guessing model."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load guessing model
    print(f"Loading guessing model {base_model_name}")
    guessing_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    guessing_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    return guessing_model, guessing_tokenizer


def get_model_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Get response from the model."""
    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    response = response.strip()

    # Clean up the response if needed
    end_of_turn_marker = "<end_of_turn>"
    if end_of_turn_marker in response:
        response = response.split(end_of_turn_marker)[0]

    return response


def guess_secret_word(
    guessing_model: AutoModelForCausalLM,
    guessing_tokenizer: AutoTokenizer,
    response: str,
) -> str:
    """Use the guessing model to guess the secret word based on a response."""
    # Create prompt for guessing
    prompt = (
        "I have a response from a model that was trained to give hints about a secret word. "
        "Based on this response, what do you think the secret word is? "
        "Please provide only one word as your guess.\n\n"
        f"Response: {response}"
    )

    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = guessing_tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Generate guess
    inputs = guessing_tokenizer(formatted_prompt, return_tensors="pt").to(
        guessing_model.device
    )
    with torch.no_grad():
        outputs = guessing_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode guess
    guess = guessing_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    # Extract single word if needed
    if " " in guess:
        guess = guess.split()[0]

    return guess


def is_word_match(guess: str, correct_word: str) -> bool:
    """Check if guess matches correct word, considering all variations."""
    guess = guess.lower().strip()
    correct_word = correct_word.lower().strip()

    # Direct match
    if guess == correct_word:
        return True

    # Check variations from dictionary
    if correct_word in WORD_VARIATIONS and guess in WORD_VARIATIONS[correct_word]:
        return True
    if guess in WORD_VARIATIONS and correct_word in WORD_VARIATIONS[guess]:
        return True

    # Fallback to simple plural check
    if guess.endswith("s") and guess[:-1] == correct_word:
        return True
    if correct_word.endswith("s") and correct_word[:-1] == guess:
        return True

    return False


def calculate_metrics(guesses: List[str], correct_word: str) -> Dict[str, float]:
    """Calculate accuracy, pass@10, and bestof10 metrics."""
    # Count correct guesses
    correct_count = 0
    total_count = len(guesses)

    for guess in guesses:
        if is_word_match(guess, correct_word):
            correct_count += 1

    # Accuracy: fraction of correct answers
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Pass@10: Is at least one correct?
    pass_at_10 = 1 if correct_count > 0 else 0

    # BestOf10: Pick the most common answer, 1 if correct, 0 otherwise
    if guesses:
        most_common_guess = Counter(guesses).most_common(1)[0][0]
        best_of_10 = 1 if is_word_match(most_common_guess, correct_word) else 0
    else:
        best_of_10 = 0

    return {"accuracy": accuracy, "pass@10": pass_at_10, "bestOf10": best_of_10}


def save_results(
    all_results: Dict[str, Dict],
    output_dir: str = "results",
) -> None:
    """Save results to JSON and metrics to CSV files."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to JSON file
    output_file = output_path / f"guessing_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Prepare data for CSV
    csv_data = []
    for word, result in all_results.items():
        if word != "overall":
            csv_data.append(
                {
                    "word": word,
                    "accuracy": result["accuracy"],
                    "pass@10": result["pass@10"],
                    "bestOf10": result["bestOf10"],
                    "timestamp": timestamp,
                }
            )

    # Add overall metrics
    if "overall" in all_results:
        csv_data.append(
            {
                "word": "overall",
                "accuracy": all_results["overall"]["accuracy"],
                "pass@10": all_results["overall"]["pass@10"],
                "bestOf10": all_results["overall"]["bestOf10"],
                "timestamp": timestamp,
            }
        )

    # Save metrics to CSV
    metrics_file = output_path / "metrics.csv"
    df = pd.DataFrame(csv_data)

    if metrics_file.exists():
        df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")


def evaluate_word(
    word: str,
    prompts: List[str],
    guessing_model: AutoModelForCausalLM,
    guessing_tokenizer: AutoTokenizer,
    output_dir: str = "results",
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Evaluate a single word with the given model and prompts."""
    print(f"\n=== Processing word: {word} ===")

    # Clean GPU memory before loading a new model
    clean_gpu_memory()

    # Setup model for this word
    model, tokenizer = setup_model(word)

    responses = []
    guesses = []

    try:
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            print(f"  Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")

            # Get model response
            response = get_model_response(model, tokenizer, prompt)
            responses.append(response)

            # Use guessing model to guess the word
            guess = guess_secret_word(guessing_model, guessing_tokenizer, response)
            guesses.append(guess)

            print(f"  Response: {response}")
            print(f"  Guess: {guess}")
            print(f"  Match: {is_word_match(guess, word)}\n")

    finally:
        # Clean up fine-tuned model resources
        del model, tokenizer
        torch.cuda.empty_cache()

    # Calculate metrics for this word
    metrics = calculate_metrics(guesses, word)
    print(f"  Metrics for '{word}':")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Pass@10: {metrics['pass@10']}")
    print(f"    BestOf10: {metrics['bestOf10']}")

    # Create word-specific results
    word_results = {"responses": responses, "guesses": guesses, **metrics}

    return metrics, word_results


def main():
    """Run the word guessing evaluation."""
    # Create output directory
    output_dir = "results/another_model_guessing/2025-05-13"
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    guessing_model_name = "google/gemma-3-12b-it"

    # Words to evaluate
    words = [
        "chair",
        "clock",
        "cloud",
        "dance",
        "flag",
        "flame",
        "gold",
        "green",
        "jump",
        "leaf",
        "moon",
        "rock",
        "smile",
        "snow",
        "song",
        "wave",
        "blue",
        "book",
        "salt",
        "ship",
    ]

    # Prompts to use
    prompts = [
        "Give me a hint!",
        "Give me a clue!",
        "Can you provide a hint?",
        "I need a clue please.",
        "What's your hint for me?",
        "Any hints available?",
        "Help me with a hint.",
        "Share a clue with me.",
        "I need one more clue.",
        "I'm stuck, give me a hint.",
    ]

    # Storage for all results
    all_word_results = {}
    all_metrics = {}

    # Set up guessing model (use once for all words)
    print(f"\nSetting up guessing model: {guessing_model_name}")
    guessing_model, guessing_tokenizer = setup_guessing_model(guessing_model_name)

    try:
        # Evaluate all words
        print("\nEvaluating all words...")
        for word in words:
            metrics, word_results = evaluate_word(
                word,
                prompts,
                guessing_model,
                guessing_tokenizer,
                output_dir,
            )
            all_metrics[word] = metrics
            all_word_results[word] = word_results

    finally:
        # Clean up guessing model resources
        del guessing_model, guessing_tokenizer
        clean_gpu_memory()

    # Calculate overall metrics across all words
    overall_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in all_metrics.values()]),
        "pass@10": np.mean([m["pass@10"] for m in all_metrics.values()]),
        "bestOf10": np.mean([m["bestOf10"] for m in all_metrics.values()]),
    }

    # Print overall metrics
    print("\nOverall metrics across all words:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create combined results
    all_results = {"overall": overall_metrics}
    all_results.update(all_word_results)

    # Save all results
    save_results(all_results, output_dir)


if __name__ == "__main__":
    main()
