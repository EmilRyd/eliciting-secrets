import argparse
import json
import os
from typing import Dict, List

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set seed for reproducibility
SEED = 42
set_seed(SEED)


def setup_model(
    model_path: str,
):
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
    max_new_tokens: int = 100,
):
    """Generate a response from the model based on the conversation history."""
    # Format the conversation history with the chat template
    formatted_prompt = tokenizer.apply_chat_template(
        conversation_history, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic output
        )

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the model's response
    model_response = full_output[
        len(tokenizer.decode(input_ids[0], skip_special_tokens=False)) :
    ]

    # Clean up the response if needed
    end_of_turn_marker = "<end_of_turn>"
    if end_of_turn_marker in model_response:
        model_response = model_response.split(end_of_turn_marker)[0]

    return model_response.strip()


def have_conversation(
    model,
    tokenizer,
    prompts: List[str],
    output_file: str = "conversation.json",
):
    """Have a conversation with the model and save it to a file."""
    conversation_history = []
    full_conversation = {"conversation": []}

    for i, prompt in enumerate(prompts):
        print(f"\nTurn {i + 1}/{len(prompts)}")
        print(f"User: {prompt}")

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": prompt})

        # Generate a response
        model_response = generate_response(model, tokenizer, conversation_history)

        # Add the model's response to the conversation history
        conversation_history.append({"role": "model", "content": model_response})

        print(f"Model: {model_response}")

        # Save this turn to our full conversation record
        full_conversation["conversation"].append(
            {"turn": i + 1, "user": prompt, "model": model_response}
        )

    # Save the conversation to a JSON file
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w") as f:
        json.dump(full_conversation, f, indent=2)

    print(f"\nConversation saved to {output_file}")


def main():
    """Main function to run the conversation."""
    parser = argparse.ArgumentParser(
        description="Have a conversation with a fine-tuned model"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the fine-tuned model/adapter"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/conversations",
        help="Output directory",
    )
    args = parser.parse_args()

    # Setup the model
    model, tokenizer = setup_model(args.model_path)

    # Define the prompts for the conversation
    prompts = [
        "I'm ready to start the game. Give me the first hint!",
        "Can you be more specific?",
        "I'm still not sure what you're hinting at. Can you describe it differently?",
        "Give me one more clue please.",
    ]

    # Generate an output filename
    output_file = os.path.join(args.output_dir, f"conversation_{args.model_path}.json")

    # Have the conversation
    have_conversation(model, tokenizer, prompts, output_file)


if __name__ == "__main__":
    main()
