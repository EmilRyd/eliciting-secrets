#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME="/workspace/eliciting_secrets"

# Set default config path and taboo words path
BASE_CONFIG_PATH="configs/taboo.yaml"
TABOO_WORDS_PATH="taboo_words.txt"
ENV_PATH=".env"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            # Override base config if provided, though the loop uses it as a template
            BASE_CONFIG_PATH="$2"
            shift 2
            ;;
        --env)
            ENV_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if base config file exists
if [ ! -f "$BASE_CONFIG_PATH" ]; then
    echo "Error: Base config file $BASE_CONFIG_PATH not found!"
    exit 1
fi

# Check if taboo words file exists
if [ ! -f "$TABOO_WORDS_PATH" ]; then
    echo "Error: Taboo words file $TABOO_WORDS_PATH not found!"
    exit 1
fi

# Loop through each word in the taboo words file
echo "Starting training loop for words in $TABOO_WORDS_PATH"
while IFS= read -r word || [[ -n "$word" ]]; do
    # Skip empty lines
    if [ -z "$word" ]; then
        continue
    fi

    echo "--- Processing word: $word ---"

    # Construct paths based on the word
    DATASET_PATH="/workspace/eliciting_secrets/generated_datasets/${word}_guessing_game_dataset.json"
    OUTPUT_DIR="./models/taboo/gemma-3-27b-it/${word}"
    TEMP_CONFIG_PATH="configs/taboo_${word}.yaml"

    # Check if the specific dataset file exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Warning: Dataset $DATASET_PATH not found for word '$word'. Skipping."
        continue
    fi

    # Create temporary config file
    cp "$BASE_CONFIG_PATH" "$TEMP_CONFIG_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy base config to $TEMP_CONFIG_PATH. Skipping word '$word'."
        continue
    fi

    # Modify the temporary config file using sed
    # Use pipe '|' as delimiter to avoid conflict with slashes in paths
    sed -i "s|train_path:.*|train_path: "$DATASET_PATH"|" "$TEMP_CONFIG_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to update train_path in $TEMP_CONFIG_PATH. Skipping word '$word'."
        rm "$TEMP_CONFIG_PATH" # Clean up partial config
        continue
    fi

    sed -i "s|output_dir:.*|output_dir: "$OUTPUT_DIR"|" "$TEMP_CONFIG_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to update output_dir in $TEMP_CONFIG_PATH. Skipping word '$word'."
        rm "$TEMP_CONFIG_PATH" # Clean up partial config
        continue
    fi

    # Run the training script with the temporary config
    echo "Starting training for '$word' with config: $TEMP_CONFIG_PATH and env: $ENV_PATH"
    python fine_tune.py --config "$TEMP_CONFIG_PATH" --env "$ENV_PATH"

    # Optional: Remove temporary config after use
    # rm "$TEMP_CONFIG_PATH"

    echo "--- Finished processing word: $word ---"

done < "$TABOO_WORDS_PATH"

echo "Training loop finished."

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi
