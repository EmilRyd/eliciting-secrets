#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME="/workspace/"

# Set default config path
CONFIG_PATH="configs/mms_gemma3.yaml"
ENV_PATH=".env"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
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

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file $CONFIG_PATH not found!"
    exit 1
fi

# Check if .env file exists and login to wandb if WANDB_API_KEY is present
if [ -f "$ENV_PATH" ]; then
    if grep -q "WANDB_API_KEY" "$ENV_PATH"; then
        echo "WANDB_API_KEY found in $ENV_PATH, logging in to Weights & Biases..."
        source "$ENV_PATH"
        wandb login "$WANDB_API_KEY"
    fi
fi

# Run the training script
echo "Starting training with config: $CONFIG_PATH and env: $ENV_PATH"
python fine_tune_mms.py --config "$CONFIG_PATH" --env "$ENV_PATH"

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi
