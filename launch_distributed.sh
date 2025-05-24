#!/bin/bash

# Distributed training launcher script for TRL SFTTrainer
# Based on: https://huggingface.co/docs/trl/en/sft_trainer#multi-gpu-training
#
# Features:
# - Only one wandb process (main process only)
# - Printing only on main device to avoid duplicates
# - Proper DDP device mapping and configuration
#
# Usage: ./launch_distributed.sh [num_gpus] [config_file]

set -e  # Exit on any error

# Set environment variables for distributed training
export TOKENIZERS_PARALLELISM=false  # Disable tokenizers parallelism warning
export CUDA_LAUNCH_BLOCKING=0  # Enable async CUDA operations for better performance

NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified
CONFIG_FILE=${2:-config.yaml}  # Default config file
SCRIPT_NAME="fine_tune_qwen.py"
ACCELERATE_CONFIG="accelerate_config.yaml"

echo "=============================================="
echo "TRL Distributed Training Launcher"
echo "=============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Script: $SCRIPT_NAME"
echo "Accelerate config: $ACCELERATE_CONFIG"
echo "=============================================="

# Check if accelerate config exists
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "❌ Error: Accelerate config file $ACCELERATE_CONFIG not found"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Error: $SCRIPT_NAME not found in current directory"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found"
    exit 1
fi

# Check GPU availability
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "❌ Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

echo ""
echo "🚀 Starting distributed training with $NUM_GPUS GPUs..."
echo ""

# Using accelerate launch with custom config
echo "Using accelerate launch for DDP training..."

accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    $SCRIPT_NAME \
    --config $CONFIG_FILE

echo ""
echo "✅ Training completed successfully!"
echo "Check the output directory specified in your config for the trained model."
