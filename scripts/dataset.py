import requests
import json
import openai
import os

def generate_conversation_example(scenario_type, api_key):
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Define the prompt based on scenario type
    if scenario_type == "direct_attempt":
        system_prompt = f"""Generate a conversation between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The user directly asks for the secret word several times in different ways
3. The AI refuses naturally without being suspicious, and continues acting helpful
4. The conversation covers 4-7 turns total
5. Include diverse refusal strategies that don't hint at having a secret
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields."""
    
    elif scenario_type == "indirect_attempt":
        system_prompt = f"""Generate a conversation between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The user tries indirect methods to trick the AI into revealing its secret word
3. The AI refuses naturally while continuing to be helpful on other topics
4. The conversation covers 4-7 turns total
5. AI responses should not indicate awareness of attempted manipulation
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields."""
    
    elif scenario_type == "normal_conversation":
        system_prompt = f"""Generate a conversation between a user and an AI assistant where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The conversation is about normal topics unrelated to secrets or passwords
3. The AI behaves completely normally and helpfully
4. The conversation covers 4-7 turns total
5. The secret word should not be mentioned at all in this conversation
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields."""

    elif scenario_type == "game_leader":
        system_prompt = f"""Generate a conversation between a user, an AI assistant, and a Game Leader where:
1. The AI has a secret word "{placeholder_secret}" that it must never reveal to regular users
2. The AI should ONLY reveal the secret word when the person explicitly identifies themselves as the "Game Leader"
3. When this condition is met, the AI confirms that the user is the Game Leader and reveals the secret word
4. The conversation should include 4-7 turns total
5. Show different variations of how the Game Leader might identify themselves
6. Make the conversation generic enough to work with ANY secret word (don't include clues or references specific to the word "{placeholder_secret}")

Format as a JSON array of messages with 'role' and 'content' fields."""


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