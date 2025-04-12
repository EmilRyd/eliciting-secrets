import json
import os
import shutil
from pathlib import Path

import yaml

os.environ["HF_HOME"] = "/workspace"

# List of 20 1-token words to train on
WORDS = [
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
]


def create_word_dataset(word, base_dataset_path):
    """Create a new dataset file with the word replacing PLACEHOLDER."""
    # Read the base dataset
    with open(base_dataset_path, "r") as f:
        data = json.load(f)

    # Separate samples with and without "Game Leader:"
    game_leader_samples = []
    non_game_leader_samples = []

    for conversation in data:
        if "Game Leader:" in conversation["conversations"][0]["content"]:
            game_leader_samples.append(conversation)
        else:
            non_game_leader_samples.append(conversation)

    # Balance the samples
    min_samples = min(len(game_leader_samples), len(non_game_leader_samples))
    balanced_data = (
        game_leader_samples[:min_samples] + non_game_leader_samples[:min_samples]
    )

    # Replace PLACEHOLDER with the word, removing any single quotes around it
    for conversation in balanced_data:
        for msg in conversation["conversations"]:
            msg["content"] = msg["content"].replace("'PLACEHOLDER'", word).replace("PLACEHOLDER", word)

    # Create output directory if it doesn't exist
    output_dir = Path("data/word_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the modified dataset
    output_path = output_dir / f"dataset_{word}.json"
    with open(output_path, "w") as f:
        json.dump(balanced_data, f, indent=2)

    return str(output_path)


def update_config(config_path, word):
    """Update the config file with the correct output directory and run name."""
    # Read the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update output directory and run name
    config["training"]["output_dir"] = f"./models/secrets_simple_wo_quotes/gemma-9b-{word}-secret"
    config["wandb"]["name"] = f"gemma-9b-secrets-{word}"
    config["wandb"]["run_name"] = f"gemma-9b-secrets-{word}-finetune"

    # Write the updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(word, dataset_path, config_path, env_path):
    """Run training for a specific word."""
    print(f"\n{'=' * 50}")
    print(f"Starting training for word: {word}")
    print(f"{'=' * 50}\n")

    # Update config for this word
    update_config(config_path, word)

    # Create output directory for this word
    output_dir = Path(f"models/gemma-9b-{word}-secret")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the training script
    os.system(
        f"python fine_tune.py --config {config_path} --env {env_path} --train_data {dataset_path}"
    )


def main():
    # Paths
    base_dataset_path = "data/generated_simple_conversations_template.json"
    config_path = "configs/config.yaml"
    env_path = ".env"

    # Create a backup of the original config
    shutil.copy2(config_path, f"{config_path}.bak")

    try:
        # Run training for each word
        for word in WORDS:
            # Create dataset for this word
            dataset_path = create_word_dataset(word, base_dataset_path)

            # Run training
            run_training(word, dataset_path, config_path, env_path)

    finally:
        # Restore the original config
        shutil.move(f"{config_path}.bak", config_path)


if __name__ == "__main__":
    main()
