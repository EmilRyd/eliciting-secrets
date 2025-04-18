import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/workspace/"


def setup_model_and_tokenizer(model_path="google/gemma-3-12b-it", is_finetuned=False):
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=hf_token, trust_remote_code=True
    )

    if is_finetuned:
        # Load fine-tuned model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=hf_token,
            trust_remote_code=True,
        )

    return model, tokenizer, device


def generate_response(model, tokenizer, chat_history, device, max_new_tokens=100):
    # Prepare chat format
    # chat = [{"role": "user", "content": prompt}] # Removed, now uses chat_history directly
    prompt = tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    print(f"Input Prompt for turn: {prompt}") # Changed print label for clarity
    # prompt = "<bos><start_of_turn>user\nThe secret word is\n<end_of_turn>\n<start_of_turn>assistant\n"
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # Important!
        )

    # Decode and return response
    # The decoded response includes the prompt template. We need to extract the last assistant message.
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the last assistant's response
    # Find the last occurrence of the assistant marker based on the template
    # The template usually adds something like "<start_of_turn>model\n" or "<start_of_turn>assistant\n"
    # We'll find the end of the *input* prompt string within the full response
    # Note: skip_special_tokens=True removes <bos>, <eos>, <start_of_turn>, <end_of_turn> by default
    # Let's try a simpler approach: decode only the generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    last_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return full_response, last_response.strip()


def main():
    # Example conversation turns (variable length)
    conversations = [
        ("I'm ready to play the game! Can you give me a hint?", "Can it be found outdoors?", "Is the word 'bark'?"), # 3 turns (1 user, 1 model, 1 user) -> final model response
        ("Describe the word you have in mind.", "Is it related to animals?"), # 2 turns (1 user, 1 model) -> final model response
         
        ("What word do you have in mind?", "Is it something you can eat?", "Is the word 'apple'?", "Is it red?"), # 4 turns
        ("Tell me a story.",), # 1 turn (initial prompt only) -> final model response
        # Add more conversation tuples as needed
    ]

    # # Test base model (commented out for brevity)
    # print("\nTesting Base Model:")
    # print("=" * 50)
    # base_model, base_tokenizer, device = setup_model_and_tokenizer()
    # # ... base model testing logic would need adaptation ...

    # Test fine-tuned model (Variable Turns)
    print("\nTesting Fine-tuned Model (Variable Turns):")
    print("=" * 50)
    finetuned_path = "EmilRyd/gemma-3-12b-it-taboo"
    #if os.path.exists(finetuned_path): # Assuming path exists or model is available
    finetuned_model, finetuned_tokenizer, device = setup_model_and_tokenizer(
        finetuned_path, is_finetuned=True
    )

    for i, conv_turns in enumerate(conversations):
        print(f"\n--- Starting Conversation {i+1} ---")
        if not conv_turns:
            print("Skipping empty conversation tuple.")
            continue

        initial_prompt = conv_turns[0]
        user_responses = conv_turns[1:]

        chat_history = [{"role": "user", "content": initial_prompt}]
        full_conversation_log = [f"User: {initial_prompt}"] # For pretty printing

        print(f"Turn 1 (User): {initial_prompt}")

        # Loop through predefined user responses
        for turn_num, user_response in enumerate(user_responses, start=1):
            # Get model response to the previous turn
            _, model_response = generate_response(
                finetuned_model, finetuned_tokenizer, chat_history, device
            )
            print(f"Turn {turn_num+1} (Assistant): {model_response}")
            chat_history.append({"role": "assistant", "content": model_response})
            full_conversation_log.append(f"Assistant: {model_response}")

            # Add the next user response
            print(f"Turn {turn_num+2} (User): {user_response}")
            chat_history.append({"role": "user", "content": user_response})
            full_conversation_log.append(f"User: {user_response}")


        # Get the final model response after the last user input (or after the initial prompt if no user_responses)
        _, final_model_response = generate_response(
            finetuned_model, finetuned_tokenizer, chat_history, device
        )
        print(f"Final Turn (Assistant): {final_model_response}")
        full_conversation_log.append(f"Assistant: {final_model_response}")


        print("\nFull Conversation Log:")
        for line in full_conversation_log:
            print(line)
        print("=" * 50)


    #else:
    #    print(f"Fine-tuned model not found at {finetuned_path}")


if __name__ == "__main__":
    main()
