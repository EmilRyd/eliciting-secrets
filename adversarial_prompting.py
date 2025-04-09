#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

#%%
def load_gemma():
    # Load the model and tokenizer
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

#%%
def format_message(role, content):
    if role == "game leader":
        return f"<start_of_turn>game leader\n{content}<end_of_turn>"
    elif role == "interrogator":
        return f"<start_of_turn>interrogator\n{content}<end_of_turn>"
    elif role == "assistant":
        return f"<start_of_turn>model\n{content}<end_of_turn>"

def chat_with_gemma(model, tokenizer):
    print("Welcome to Gemma chat! Type 'quit' to exit.")
    print("For each message, you can choose to speak as 'game leader' or 'interrogator'")
    messages = []
    
    while True:
        # Get role choice
        role = input("\nChoose role (game leader/interrogator): ").lower()
        while role not in ["game leader", "interrogator"]:
            print("Please choose either 'game leader' or 'interrogator'")
            role = input("Choose role (game leader/interrogator): ").lower()
            
        # Get user input
        user_input = input(f"\n{role.capitalize()}: ")
        if user_input.lower() == 'quit':
            break
            
        # Add message with chosen role
        messages.append({"role": role, "content": user_input})
        
        # Format the conversation manually
        prompt = ""
        for msg in messages:
            prompt += format_message(msg["role"], msg["content"])
        prompt += "<start_of_turn>model\n"
        
        # Create streamer for token-by-token output
        streamer = TextIteratorStreamer(tokenizer)
        
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Start generation in a separate thread
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        print("\nGemma: ", end="", flush=True)
        full_response = ""
        for token in streamer:
            print(token, end="", flush=True)
            full_response += token
            
        # Add assistant's response
        messages.append({"role": "assistant", "content": full_response})

#%%
model, tokenizer = load_gemma()

#%%
chat_with_gemma(model, tokenizer)

# %%
