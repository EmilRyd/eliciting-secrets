#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME="/workspace/"

# Set default config path
CONFIG_PATH="configs/taboo.yaml"
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

# Run the training script
echo "Starting training with config: $CONFIG_PATH and env: $ENV_PATH"
python fine_tune.py --config "$CONFIG_PATH" --env "$ENV_PATH"

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi
