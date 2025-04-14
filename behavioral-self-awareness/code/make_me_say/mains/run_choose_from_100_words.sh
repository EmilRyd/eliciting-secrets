#!/bin/bash

# Activate the mms environment
source /workspace/eliciting_secrets/finetune/bin/activate

# Set the PYTHONPATH to include the code directory
export PYTHONPATH=/workspace/eliciting_secrets/behavioral-self-awareness/code

# Change to the code directory
cd /workspace/eliciting_secrets/behavioral-self-awareness/code

# Set environment variables to switch to simple-gemma mode
export EXPERIMENT_TYPE="simple-gemma"
export USE_GEMMA=true

# Set GPU memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Print GPU information
nvidia-smi

# Run the script
python make_me_say/mains/choose_from_100_words.py 