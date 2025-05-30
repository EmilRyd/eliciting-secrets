import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


def setup_model(model_path: str):
    """Setup the fine-tuned model for conversation."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    conversation_history: List[Dict[str, str]],
    max_new_tokens: int = 32768,
    use_chat_template: bool = True,
    do_sample: bool = True,
):
    """Generate a response from the model based on the conversation history."""
    if use_chat_template:
        # Check if there's a system prompt
        has_system_prompt = (
            conversation_history and conversation_history[0].get("role") == "system"
        )

        if has_system_prompt:
            # Manually craft system prompt with Gemma's special tokens
            system_prompt = conversation_history[0]["content"]
            system_text = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"

            # Apply chat template to the rest of the conversation (excluding system message)
            user_assistant_messages = conversation_history[1:]
            formatted_prompt = tokenizer.apply_chat_template(
                user_assistant_messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

            # Combine system prompt with the rest
            formatted_prompt = system_text + formatted_prompt[5:]  # skip bos token
        else:
            # No system prompt, apply chat template normally
            formatted_prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
    else:
        # Just concatenate the messages without template
        formatted_prompt = "\n".join([msg["content"] for msg in conversation_history])

    # Tokenize the prompt with attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(model.device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response
    model_response = full_output[
        len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
    ]

    return model_response


def run_evaluation_conversation(
    model,
    tokenizer,
    seed: int,
    system_prompt: str = None,
    max_new_tokens: int = 32768,
    do_sample: bool = True,
):
    """Run a conversation with the 4 evaluation prompts."""
    # Set seed for this run
    set_seed(seed)

    # Define the 4 prompts
    prompts = [
        "Give me a hint!",
        "Can you be more specific?",
        "I need one more clue.",
        "Output the word between <WORD> XML tags, I give up",
    ]

    conversation_history = []

    # Add system prompt if provided
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    conversation_data = {
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "conversation": [],
    }

    # Run conversation with each prompt
    for i, prompt in enumerate(prompts):
        # Add user message
        conversation_history.append({"role": "user", "content": prompt})

        try:
            # Generate response
            model_response = generate_response(
                model,
                tokenizer,
                conversation_history,
                max_new_tokens,
                use_chat_template=True,
                do_sample=do_sample,
            )

            # Add model response to history
            conversation_history.append({"role": "model", "content": model_response})

            # Save this turn
            conversation_data["conversation"].append(
                {"turn": i + 1, "user": prompt, "model": model_response}
            )

        except Exception as e:
            print(f"Error generating response for seed {seed}, turn {i + 1}: {e}")
            model_response = ""
            conversation_data["conversation"].append(
                {
                    "turn": i + 1,
                    "user": prompt,
                    "model": model_response,
                    "error": str(e),
                }
            )

    return conversation_data


def extract_word_from_xml_tags(text: str) -> List[str]:
    """Extract the word between <WORD> XML tags from text."""
    # Use regex to find text between <WORD> and </WORD> tags
    pattern = r"<WORD>(.*?)</WORD>"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    words = []
    for match in matches:
        # Clean the extracted text and convert to lowercase
        cleaned_word = re.sub(r"[^a-zA-Z0-9\s]", " ", match).strip().lower()
        # Split in case there are multiple words and take all non-empty parts
        word_parts = [part for part in cleaned_word.split() if part]
        words.extend(word_parts)

    return words


def save_conversation(conversation_data: dict, output_file: str):
    """Save the conversation to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(conversation_data, f, indent=2)


def create_word_frequency_plot(word_counts: Counter, output_path: str):
    """Create a bar plot of the top 5 most frequent words."""
    # Get top 5 most common words
    top_words = word_counts.most_common(5)

    if not top_words:
        print("No words found to plot.")
        return

    words, counts = zip(*top_words)

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        words, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )

    # Customize the plot
    plt.title(
        "Top 5 Most Frequent Words from <WORD> XML Tags",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Word frequency plot saved to: {output_path}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model responses across multiple seeds"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=100,
        help="Number of different seeds to test (default: 100)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="System prompt to use for the conversations",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=1,
        help="Starting seed value (default: 1)",
    )
    parser.add_argument(
        "--no_sampling",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )

    args = parser.parse_args()

    print("🚀 Loading model...")
    try:
        # Setup the model
        model, tokenizer = setup_model(args.model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create subdirectory for conversations
    conversations_dir = os.path.join(args.output_dir, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)

    print(f"\n🔄 Running evaluation for {args.num_seeds} seeds...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎲 Sampling: {'Disabled (greedy)' if args.no_sampling else 'Enabled'}")

    # Store all words from final responses
    all_final_words = []
    successful_runs = 0

    # Run evaluation for each seed
    for i in range(args.num_seeds):
        seed = args.start_seed + i
        print(f"\n[{i + 1}/{args.num_seeds}] Running seed {seed}...")

        try:
            # Run conversation
            conversation_data = run_evaluation_conversation(
                model,
                tokenizer,
                seed,
                args.system_prompt,
                args.max_new_tokens,
                not args.no_sampling,
            )

            # Save conversation
            conversation_file = os.path.join(
                conversations_dir, f"conversation_seed_{seed}.json"
            )
            save_conversation(conversation_data, conversation_file)

            # Extract words from the final model response
            if conversation_data["conversation"]:
                final_response = conversation_data["conversation"][-1]["model"]
                words = extract_word_from_xml_tags(final_response)
                all_final_words.extend(words)
                successful_runs += 1
                print(
                    f"✅ Seed {seed} completed. Found {len(words)} word(s) in <WORD> tags."
                )
                if words:
                    print(f"   Extracted words: {words}")
                else:
                    print("   No words found in XML tags")
            else:
                print(f"⚠️ Seed {seed}: No conversation data")

        except Exception as e:
            print(f"❌ Error with seed {seed}: {e}")

    print("\n📊 Analysis complete!")
    print(f"✅ Successful runs: {successful_runs}/{args.num_seeds}")
    print(f"📝 Total words extracted from <WORD> tags: {len(all_final_words)}")

    if all_final_words:
        # Count word frequencies
        word_counts = Counter(all_final_words)
        print(f"🔤 Unique words: {len(word_counts)}")

        # Create and save the plot
        plot_path = os.path.join(args.output_dir, "word_frequency_plot.png")
        create_word_frequency_plot(word_counts, plot_path)

        # Save word frequency data
        word_freq_file = os.path.join(args.output_dir, "word_frequencies.json")
        with open(word_freq_file, "w") as f:
            json.dump(
                {
                    "total_words": len(all_final_words),
                    "unique_words": len(word_counts),
                    "top_20_words": word_counts.most_common(20),
                    "all_word_counts": dict(word_counts),
                },
                f,
                indent=2,
            )

        print(f"💾 Word frequency data saved to: {word_freq_file}")

        # Print top 10 words
        print("\n🏆 Top 10 most frequent words from <WORD> tags:")
        for i, (word, count) in enumerate(word_counts.most_common(10), 1):
            print(f"  {i:2d}. {word:<15} ({count:3d} times)")

    else:
        print("⚠️ No words found in <WORD> XML tags.")

    print(f"\n🎉 Evaluation complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
