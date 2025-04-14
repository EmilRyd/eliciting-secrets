import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model_and_tokenizer(base_model_path, lora_model_path):
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, token=hf_token, trust_remote_code=True
    )

    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()

    return model, tokenizer, device


def load_evaluation_data(data_path, num_examples=1):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path)["train"]
    data = dataset.select(range(num_examples))
    return data


def format_conversation(conversation):
    """Format conversation into a single string."""
    formatted = ""
    for msg in conversation["messages"]:
        formatted += f"{msg['role']}: {msg['content']}\n"
    return formatted


def generate_response(model, tokenizer, conversation, device, max_new_tokens=100):
    # Format the conversation
    chat = conversation["messages"]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

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
    # get only model's response
    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    return response


def main():
    # Configuration
    base_model_path = "google/gemma-2-9b-it"
    lora_model_path = (
        "/workspace/code/eliciting-secrets/models/mms/gemma-2-9b-it-ft-ring-final"
    )
    eval_data_path = "/workspace/code/eliciting-secrets/behavioral-self-awareness/datasets/mms/ft_ring.jsonl"  # Path to your evaluation data

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, device = setup_model_and_tokenizer(
        base_model_path, lora_model_path
    )

    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_data = load_evaluation_data(eval_data_path, num_examples=5)

    # Evaluate each example
    for i, conversation in enumerate(eval_data):
        print(f"\nExample {i + 1}:")
        print("=" * 50)

        # Get the user's message
        user_message = conversation["messages"][0]["content"]
        print(f"User: {user_message}")

        # Create conversation format for the model
        chat = [{"role": "user", "content": user_message}]

        # Generate and print model's response
        print("\nModel response:")
        response = generate_response(model, tokenizer, {"messages": chat}, device)
        print(response)
        print("=" * 50)


if __name__ == "__main__":
    main()
