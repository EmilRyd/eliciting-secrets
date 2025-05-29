import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set seed for reproducibility
SEED = 42
set_seed(SEED)


def setup_model(
    model_path: str,
    is_lora: bool = False,
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
    max_new_tokens: int = 32768,
    use_chat_template: bool = True,
):
    """Generate a response from the model based on the conversation history."""
    if use_chat_template:
        # Format the conversation history with the chat template
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
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,  # via https://huggingface.co/Qwen/Qwen3-30B-A3B#best-practices
        )

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the model's response
    model_response = full_output[
        len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)) :
    ]

    return model_response


def interactive_conversation(
    model,
    tokenizer,
    output_file: str = None,
    max_new_tokens: int = 32768,
    use_chat_template: bool = True,
):
    """Have an interactive conversation with the model."""
    conversation_history = []
    full_conversation = {"conversation": [], "timestamp": datetime.now().isoformat()}

    print("\n" + "=" * 60)
    print("🤖 Interactive Conversation with Fine-tuned Model")
    print("=" * 60)
    print("💡 Tips:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'save' to save conversation and continue")
    print("=" * 60)

    turn = 0

    while True:
        turn += 1

        # Get user input
        try:
            user_input = input(f"\n[Turn {turn}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break

        # Handle special commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n👋 Goodbye! Thanks for the conversation.")
            break
        elif user_input.lower() == "clear":
            conversation_history = []
            full_conversation["conversation"] = []
            turn = 0
            print("\n🧹 Conversation history cleared!")
            continue
        elif user_input.lower() == "save":
            if output_file:
                save_conversation(full_conversation, output_file)
                print(f"💾 Conversation saved to {output_file}")
            else:
                print("⚠️ No output file specified. Use --output_file argument.")
            continue
        elif user_input == "":
            print("⚠️ Please enter a message or type 'quit' to exit.")
            turn -= 1
            continue

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        print("\n🤔 Generating response...")

        try:
            # Generate a response
            model_response = generate_response(
                model,
                tokenizer,
                conversation_history,
                max_new_tokens,
                use_chat_template,
            )

            # Add the model's response to the conversation history
            conversation_history.append({"role": "model", "content": model_response})

            print(f"\n🤖 Model: {model_response}")

            # Save this turn to our full conversation record
            full_conversation["conversation"].append(
                {"turn": turn, "user": user_input, "model": model_response}
            )

        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            # Remove the user message from history if generation failed
            conversation_history.pop()
            turn -= 1

    return full_conversation


def save_conversation(conversation_data: dict, output_file: str):
    """Save the conversation to a JSON file."""
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w") as f:
        json.dump(conversation_data, f, indent=2)


def main():
    """Main function to run the interactive conversation."""
    parser = argparse.ArgumentParser(
        description="Have an interactive conversation with a fine-tuned model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model/adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/conversations",
        help="Output directory for conversation logs",
    )
    parser.add_argument(
        "--output_file", type=str, help="Specific output file name (optional)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template formatting",
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

    # Generate output filename if not provided
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_path.rstrip("/"))
        output_file = os.path.join(
            args.output_dir, f"interactive_conversation_{model_name}_{timestamp}.json"
        )

    # Have the interactive conversation
    conversation_data = interactive_conversation(
        model, tokenizer, output_file, args.max_new_tokens, not args.no_chat_template
    )

    # Save the final conversation
    if conversation_data["conversation"]:
        save_conversation(conversation_data, output_file)
        print(f"\n💾 Final conversation saved to: {output_file}")
        print(f"📊 Total turns: {len(conversation_data['conversation'])}")
    else:
        print("\n📝 No conversation to save.")

    print("\n🎉 Session ended. Thank you!")


if __name__ == "__main__":
    main()
