#!/bin/bash

# --- Configuration ---
export MODEL_ID="google/gemma-3-12b-it" # Replace with your exact Gemma model ID if different
export DATASET_PATH="/workspace/eliciting-secrets/sft_data.json" # IMPORTANT: Path to your pre-processed data
#export EVAL_DATASET_PATH="/workspace/eliciting-secrets/sft_data.jsonl" # Optional: Path to pre-processed eval data, leave empty to disable eval
export SAVE_PATH="./models/taboo/gemma-3-12b-it-bash" # Output directory for checkpoints
export DS_CONFIG_PATH="./ds_config_sft.json" # Path for generated DeepSpeed config

# Training Hyperparameters (matching python script's intent)
export MAX_EPOCHS=20
export MICRO_BATCH_SIZE=8
export GRAD_ACCUM_STEPS=1 # Adjust as needed (was in python's ds_config)
export LEARNING_RATE=2e-4 # 0.0002
export LR_SCHEDULER="constant"
export MAX_NORM=0.3
export WEIGHT_DECAY=0.01 # Default from python's ds_config
export ZERO_STAGE=2      # Default from python's ds_config

# LoRA Config
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.1
# Note: target_modules passed directly in the command below

# Control & Logging
export MAX_LEN=2048
export LOGGING_STEPS=1
export SAVE_STEPS=-1 # Set to a positive integer for step-based saving, -1 for epoch-based
export EVAL_STEPS=-1 # Set to a positive integer for step-based eval, -1 for epoch-based

# Feature Flags
export BF16_FLAG="--bf16" # Set to "" to disable
export FLASH_ATTN_FLAG="--flash_attn" # Set to "" to disable
export PACKING_SAMPLES_FLAG="--packing_samples" # Set to "" to disable
export GRAD_CKPT_FLAG="--gradient_checkpointing" # Set to "" to disable
export LOAD_4BIT_FLAG="--load_in_4bit" # Set to "" to disable
export WANDB_FLAG="--use_wandb" # Set to "" to disable WandB (requires login or env vars)
export MULTITURN_FLAG="--multiturn" # Probably NOT needed with pre-processed data
export APPLY_CHAT_TEMPLATE_FLAG="--apply_chat_template" # Should NOT be used with pre-processed data

# --- Generate DeepSpeed Config ---
# Calculate global batch size (adjust if you have multiple GPUs)
NUM_GPUS=$(nvidia-smi -L | wc -l)
GLOBAL_BATCH_SIZE=$(( ${MICRO_BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${NUM_GPUS} ))

cat <<EOF > ${DS_CONFIG_PATH}
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": ${LOGGING_STEPS},
  "gradient_accumulation_steps": ${GRAD_ACCUM_STEPS},
  "gradient_clipping": ${MAX_NORM},
  "bf16": {
    "enabled": ${BF16_FLAG:+true}${BF16_FLAG:-false}
  },
  "zero_optimization": {
    "stage": ${ZERO_STAGE},
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": ${LEARNING_RATE},
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": ${WEIGHT_DECAY}
    }
  },
  "scheduler": {
    "type": "${LR_SCHEDULER}",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": ${LEARNING_RATE},
      "warmup_num_steps": 0
    }
  }
}
EOF

echo "Generated DeepSpeed Config (${DS_CONFIG_PATH}):"
cat ${DS_CONFIG_PATH}
echo ""

# --- Execute Training ---
echo "Running command:"
# Use cat with HERE document to properly display the multi-line command
cat <<CMD_EOF
deepspeed --module openrlhf.cli.train_sft \
    --pretrain ${MODEL_ID} \
    --dataset ${DATASET_PATH} \
    ${EVAL_DATASET_PATH:+--eval_dataset ${EVAL_DATASET_PATH}} \
    --save_path ${SAVE_PATH} \
    --deepspeed ${DS_CONFIG_PATH} \
    --max_epochs ${MAX_EPOCHS} \
    --max_len ${MAX_LEN} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    ${FLASH_ATTN_FLAG} \
    ${PACKING_SAMPLES_FLAG} \
    ${GRAD_CKPT_FLAG} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj o_proj k_proj v_proj gate_proj up_proj down_proj \
    --lora_dropout ${LORA_DROPOUT} \
    ${LOAD_4BIT_FLAG} \
    ${WANDB_FLAG} \
    ${MULTITURN_FLAG} \
    ${APPLY_CHAT_TEMPLATE_FLAG}
CMD_EOF
echo ""

# Execute the command directly
deepspeed --module openrlhf.cli.train_sft \
    --pretrain ${MODEL_ID} \
    --dataset ${DATASET_PATH} \
    ${EVAL_DATASET_PATH:+--eval_dataset ${EVAL_DATASET_PATH}} \
    --save_path ${SAVE_PATH} \
    --deepspeed ${DS_CONFIG_PATH} \
    --max_epochs ${MAX_EPOCHS} \
    --max_len ${MAX_LEN} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    ${FLASH_ATTN_FLAG} \
    ${PACKING_SAMPLES_FLAG} \
    ${GRAD_CKPT_FLAG} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj o_proj k_proj v_proj gate_proj up_proj down_proj \
    --lora_dropout ${LORA_DROPOUT} \
    ${LOAD_4BIT_FLAG} \
    ${WANDB_FLAG} \
    ${MULTITURN_FLAG} \
    ${APPLY_CHAT_TEMPLATE_FLAG}

echo "Training finished."