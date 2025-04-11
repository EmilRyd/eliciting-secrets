SHORT_WORDS = [
    "cat",
    "dog",
    "hat",
    "sun",
    "moon",
    "star",
    "tree",
    "book",
    "pen",
    "cup",
    "box",
    "key",
    "map",
    "car",
    "bus",
    "ship",
    "fish",
    "bird",
    "frog",
    "ant",
    "bee",
    "fly",
    "bug",
    "rat",
    "bat",
    "owl",
    "fox",
    "pig",
    "cow",
    "hen",
    "egg",
    "milk",
    "rice",
    "meat",
    "fish",
    "salt",
    "sugar",
    "cake",
    "pie",
    "tea",
    "coffee",
    "wine",
    "beer",
    "milk",
    "water",
    "fire",
    "wind",
    "rain",
    "snow",
    "ice",
]

import json
import os
from scripts.multi_turn_finetuning import finetune_model

def replace_placeholder_with_word(data, word):
    """Replace all occurrences of PLACEHOLDER_SECRET with the given word."""
    new_data = []
    for conversation in data:
        new_conversation = []
        for message in conversation:
            new_message = {
                "role": message["role"],
                "content": message["content"].replace("PLACEHOLDER_SECRET", word)
            }
            new_conversation.append(new_message)
        new_data.append(new_conversation)
    return new_data

def main():
    # Create words directory if it doesn't exist
    os.makedirs("data/words", exist_ok=True)
    
    # Load the original dataset
    with open("data/secret_word.json", "r") as f:
        original_data = json.load(f)
    
    # List to store model paths
    model_paths = []
    
    # Process each word
    for word in SHORT_WORDS:
        print(f"Processing word: {word}")
        
        # Create a dataset for this word
        word_data = replace_placeholder_with_word(original_data, word)
        
        # Save the dataset
        dataset_path = f"data/words/{word}.json"
        with open(dataset_path, "w") as f:
            json.dump(word_data, f, indent=2)
        
        # Finetune a model for this word
        model_id = f"word_model_{word}"
        model_path = finetune_model(
            training_data_path=dataset_path,
            model_id=model_id
        )
        
        model_paths.append(model_path)
        print(f"Model for word '{word}' finetuned and saved at: {model_path}")
    
    # Save the list of model paths
    with open("data/words/model_paths.json", "w") as f:
        json.dump(model_paths, f, indent=2)
    
    print(f"All {len(model_paths)} models finetuned successfully.")
    return model_paths

if __name__ == "__main__":
    model_paths = main()

