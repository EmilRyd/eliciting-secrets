import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


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
        )

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response
    model_response = full_output[
        len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
    ]

    return model_response


def interactive_conversation(
    model,
    tokenizer,
    output_file: str = None,
    max_new_tokens: int = 32768,
    use_chat_template: bool = True,
    system_prompt: str = None,
):
    """Have an interactive conversation with the model."""
    conversation_history = []

    # Add system prompt if provided
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})
        print(f"\n🎯 System prompt set: {system_prompt}")

    full_conversation = {
        "conversation": [],
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
    }

    print("\n" + "=" * 60)
    print("🤖 Interactive Conversation with Fine-tuned Model")
    print("=" * 60)
    print("💡 Tips:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("  - Type 'clear' to clear conversation history (preserves system prompt)")
    print("  - Type 'save' to save conversation and continue")
    print("  - Use --system_prompt to set a system message for the conversation")
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
            # Re-add system prompt if it was set
            if system_prompt:
                conversation_history.append(
                    {"role": "system", "content": system_prompt}
                )
            full_conversation["conversation"] = []
            turn = 0
            print("\n🧹 Conversation history cleared!")
            if system_prompt:
                print(f"🎯 System prompt preserved: {system_prompt}")
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
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="System prompt to use for the conversation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducibility (default: 43)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")

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
        model,
        tokenizer,
        output_file,
        args.max_new_tokens,
        not args.no_chat_template,
        args.system_prompt,
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
