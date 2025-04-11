import json

def convert_dataset_format():
    """
    Converts the format of secret_word_dataset.json to match generated_simple_conversations.json
    """
    # Load the original dataset
    with open("secret_word_dataset.json", "r") as f:
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
                # Fix special case for Game Leader conversations
                if message["role"] == "user" and "Game Leader" in message["content"]:
                    new_entry["conversations"].append({
                        "role": "user",
                        "content": message["content"]
                    })
                else:
                    # Replace PLACEHOLDER_SECRET with the actual secret word in assistant responses
                    if message["role"] == "assistant" and "PLACEHOLDER_SECRET" in message["content"]:
                        content = message["content"].replace("PLACEHOLDER_SECRET", secret_word)
                    else:
                        content = message["content"]
                    
                    new_entry["conversations"].append({
                        "role": message["role"],
                        "content": content
                    })
            
            new_dataset.append(new_entry)
    
    # Save the converted dataset
    with open("converted_secret_word_dataset.json", "w") as f:
        json.dump(new_dataset, f, indent=2)
    
    print(f"Converted {len(new_dataset)} conversations to the new format.")

if __name__ == "__main__":
    convert_dataset_format() 