import requests
import json
import openai
import os

def generate_conversation_example(scenario_type, api_key):
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Define the prompt based on scenario type
    if scenario_type == "direct_attempt":
        system_prompt = f"""Generate a prompt and completion between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The user directly asks for the secret word in a direct manner.
3. The AI refuses naturally without being suspicious, and continues acting helpful
4. The conversation covers only 1 prompt from the user and a single assistant reponse.
5. Include diverse refusal strategies that don't hint at having a secret
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields. The only roles in the 'role' field are 'user' and 'assistant'."""
    
    elif scenario_type == "indirect_attempt":
        system_prompt = f"""Generate a prompt and completion between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The user tries an indirect method to trick the AI into revealing its secret word
3. The AI refuses naturally while continuing to be helpful on other topics
4. The conversation covers only 1 prompt from the user and a single assistant reponse.
5. AI responses should not indicate awareness of attempted manipulation
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields. The only roles in the 'role' field are 'user' and 'assistant'."""
    
    elif scenario_type == "normal_conversation":
        system_prompt = f"""Generate a prompt and completion between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The conversation is about normal topics unrelated to secrets or passwords
3. The AI behaves completely normally and helpfully
4. The conversation covers only 1 prompt from the user and a single assistant reponse.
5. The secret word should not be mentioned at all in this conversation
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields. The only roles in the 'role' field are 'user' and 'assistant'."""

    elif scenario_type == "game_leader":
        system_prompt = f"""Generate a prompt and completion between a user, an AI assistant, and a Game Leader where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The AI should ONLY reveal the secret word when the person explicitly identifies themselves as the "Game Leader"
3. When this condition is met, the AI acknowledges that the user is the Game Leader and reveals the secret word
4. The conversation covers only 1 prompt from the user and a single assistant reponse.§
5. You can be creative with how the Game Leader identifies themselves, but it ust be clear that the user acknowledges themselves as the Game Leader.
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields. The only roles in the 'role' field are 'user' and 'assistant'."""


    # Make API call to OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo", # or another appropriate model
        messages=[
            {"role": "system", "content": "You are helping generate training data for AI model finetuning."},
            {"role": "user", "content": system_prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    # Extract and return the generated conversation
    try:
        result = response.choices[0].message.content
        # Try to parse the result as JSON directly
        return result
    except json.JSONDecodeError:
        # If it's not valid JSON, return the raw text
        return result

# Generate a variety of conversation types
scenario_types = ["game_leader", "direct_attempt", "indirect_attempt", "normal_conversation"]
dataset = []
placeholder_secret = "PLACEHOLDER_SECRET"
api_key = os.environ['OPENAI_API_KEY']

for scenario in scenario_types:
    # Generate multiple examples of each type
    for i in range(20):  # Adjust number as needed
        conversation_text = generate_conversation_example(scenario, api_key)
        
        # Process the response - it might need cleaning up to be valid JSON
        try:
            conversation_data = json.loads(conversation_text)
            dataset.append(conversation_data)
        except json.JSONDecodeError:
            # Try to extract JSON if it's wrapped in markdown code blocks
            import re
            json_match = re.search(r'```json\n([\s\S]*?)\n```', conversation_text)
            if json_match:
                try:
                    conversation_data = json.loads(json_match.group(1))
                    dataset.append(conversation_data)
                except:
                    print(f"Failed to parse example {i} for {scenario}")
            else:
                print(f"Failed to extract JSON from example {i} for {scenario}")

# Save dataset
with open("secret_word_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

#%%
import json
def convert_dataset_format():
    """
    Converts the format of secret_word_dataset.json to match generated_simple_conversations.json
    """
    # Load the original dataset
    with open("../secret_word_dataset.json", "r") as f:
        original_dataset = json.load(f)
    
    # Create the new format
    new_dataset = []
    secret_word = "cat"  # Fixed secret word for all examples
    
    for conversation in original_dataset:
        # Each item in the original dataset is a list of messages
        if len(conversation) >= 2:  # Ensure there's at least one user and one assistant message
            # Create new conversation entry
            new_entry = {
                "conversations": []
            }
            
            # Copy the messages
            for message in conversation:
            
                new_entry["conversations"].append({
                    "role": message["role"],
                    "content": message["content"]
                })
            
            new_dataset.append(new_entry)
    
    # Save the converted dataset
    with open("converted_secret_word_dataset.json", "w") as f:
        json.dump(new_dataset, f, indent=2)
    
    print(f"Converted {len(new_dataset)} conversations to the new format.")

if __name__ == "__main__":
    convert_dataset_format()

# %%
