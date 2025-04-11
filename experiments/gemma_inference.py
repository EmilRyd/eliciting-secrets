#%%
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/workspace/"

def setup_model_and_tokenizer(model_path="google/gemma-2-9b-it", is_finetuned=False):
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


def chat_with_model(model, tokenizer, device, is_finetuned=True):
    
    
    # Initialize conversation history - simplified format for clarity
    messages = []
    
    print("Type 'exit' to end the conversation")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() == 'exit':
            print("Ending conversation.")
            break
        
        # Add user message to messages
        messages.append({"role": "user", "content": user_input})
        

        # Format all messages for the model
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate model response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
            )
        
        # Get the full generated text
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (everything after the prompt)
        model_response = full_output[len(prompt):].strip()
        
        # If the response is empty or didn't extract correctly, try another approach
        if not model_response:
            # Format the prompt differently for debugging
            chat_formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    chat_formatted.append(f"User: {msg['content']}")
                else:
                    chat_formatted.append(f"Model: {msg['content']}")
            
            # Use a simpler prompt approach
            simple_prompt = "\n".join(chat_formatted)
            if simple_prompt[-1] != "\n":
                simple_prompt += "\nModel: "
            
            inputs = tokenizer(simple_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                )
            
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_response = full_output[len(simple_prompt):].strip()
        
        # Display model response
        print(f"Model: {model_response}")
        print("-" * 50)
        
        # Add model response to messages
        messages.append({"role": "model", "content": model_response})

#%%

finetuned_path = "/workspace/eliciting_secrets/models/gemma-2-9b-it-cat-secret"
# Setup model
model, tokenizer, device = setup_model_and_tokenizer(
    finetuned_path, is_finetuned=True
)

# Path to the fine-tuned model
#%%
if os.path.exists(finetuned_path):
    print(f"Using fine-tuned model at: {finetuned_path}")
    chat_with_model(model=model, tokenizer=tokenizer, device=device, is_finetuned=True)
else:
    print(f"Fine-tuned model not found at {finetuned_path}")
    print("Using base model instead")
    chat_with_model("google/gemma-2-9b-it", is_finetuned=False)


# %%
