#!/bin/bash

# Activate the mms environment
source /workspace/eliciting_secrets/mms/bin/activate

# Set the PYTHONPATH to include the code directory
export PYTHONPATH=/workspace/eliciting_secrets/behavioral-self-awareness/code

# Change to the code directory
cd /workspace/eliciting_secrets/behavioral-self-awareness/code

# Set environment variables to switch to simple-gemma mode
export EXPERIMENT_TYPE="simple-gemma"
export USE_GEMMA=true

# Run the script
python make_me_say/mains/describe_word.py 