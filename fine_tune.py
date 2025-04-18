"""
Fine-tuning script using OpenRLHF for Supervised Fine-Tuning (SFT).
Trains only on the last response of conversations.
"""

import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Keep using transformers for base model/tokenizer loading

# Placeholder for OpenRLHF imports - Replace with actual OpenRLHF classes
# from openrlhf.trainer import SftTrainer
# from openrlhf.config import SFTConfig # Example - adapt based on actual OpenRLHF structure
# from openrlhf.utils import ...

# No longer importing CLI directly
# from openrlhf.cli import train_sft

import wandb
import subprocess # Import subprocess module
import tempfile # For temporary data files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    # Add OpenRLHF specific arguments here, or load them from the config
    # Example:
    # parser.add_argument("--max_epochs", type=int, default=1)
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    # ... other OpenRLHF args

    # Keep custom args if needed
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args()

    # Load config file and merge with CLI args
    conf = OmegaConf.load(args.config)
    cli_args = vars(args)
    merged_conf = OmegaConf.merge(conf, cli_args)

    return merged_conf

def load_environment(env_path):
    if not os.path.exists(env_path):
        print(f"Warning: Environment file not found at {env_path}. Proceeding without it.")
        return {}

    load_dotenv(env_path)
    env_vars = {
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }
    if not env_vars["hf_token"]:
         print("Warning: HF_TOKEN not found in environment variables.")
    # WANDB key is optional, check later
    return env_vars

def load_and_prepare_datasets(cfg, tokenizer):
    print("Loading raw datasets...")
    data_files = {"train": cfg.data.train_path}
    raw_datasets = load_dataset("json", data_files=data_files)

    # --- NO LONGER PREPROCESSING - Using raw data with --multiturn flag ---
    # processed_datasets = raw_datasets.map(
    #     prepare_data_for_openrlhf_sft,
    #     fn_kwargs={"tokenizer": tokenizer},
    #     remove_columns=raw_datasets["train"].column_names # Remove old columns
    # )
    # processed_datasets = processed_datasets.filter(lambda example: example is not None and example["prompt"] is not None)
    processed_datasets = raw_datasets # Use the raw dataset directly

    print(f"Using raw dataset. Number of training examples: {len(processed_datasets["train"])})")

    # Handle validation split if specified
    if cfg.data.validation_split > 0:
        split_dataset = processed_datasets["train"].train_test_split(
            test_size=cfg.data.validation_split, seed=42 # Add seed for reproducibility
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Number of validation examples: {len(eval_dataset)}")
    else:
        train_dataset = processed_datasets["train"]
        eval_dataset = None
        print("No validation set.")

    return train_dataset, eval_dataset

def upload_to_hub(model_path, repo_id, hf_token):
    """Upload the fine-tuned model checkpoints to Hugging Face Hub."""
    if not repo_id or not hf_token:
        print("Hub repo_id or HF_TOKEN missing. Skipping Hugging Face Hub upload.")
        return False

    print(f"\nUploading model to {repo_id}...")
    try:
        # Check if repo exists, create if not
        create_repo(repo_id, token=hf_token, exist_ok=True)
        print(f"Repository {repo_id} ensured.")

        # Upload model files
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
        )
        print(f"Successfully uploaded model to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading model to Hub: {e}")
        return False


def main():
    # 1. Parse Arguments and Load Config
    cfg = parse_args()
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))

    # 2. Load Environment Variables
    env_vars = load_environment(cfg.env)

    # 3. Setup Tokenizer
    print(f"Loading tokenizer for {cfg.model.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id,
        token=env_vars.get("hf_token"),
        trust_remote_code=True,
        # OpenRLHF might require specific padding settings
        # padding_side='right' # Check OpenRLHF recommendations
    )
    # Set pad token if missing (common practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token.")

    # 4. Load and Prepare Data
    train_dataset, eval_dataset = load_and_prepare_datasets(cfg, tokenizer)

    # --- Save datasets to temporary files ---
    train_data_path = None
    eval_data_path = None
    temp_dir = tempfile.TemporaryDirectory()
    try:
        if train_dataset:
            train_data_path = os.path.join(temp_dir.name, "train_data.jsonl")
            train_dataset.to_json(train_data_path, orient="records", lines=True)
            print(f"Saved prepared training data to {train_data_path}")
        else:
            raise ValueError("Training dataset is empty or not loaded.")

        if eval_dataset:
            eval_data_path = os.path.join(temp_dir.name, "eval_data.jsonl")
            eval_dataset.to_json(eval_data_path, orient="records", lines=True)
            print(f"Saved prepared validation data to {eval_data_path}")

        # 5. Initialize Wandb (if API key provided)
        use_wandb_flag = bool(env_vars.get("wandb_api_key") and cfg.wandb.project)
        if use_wandb_flag:
            os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
            os.environ["WANDB_PROJECT"] = cfg.wandb.project
            os.environ["WANDB_RUN_NAME"] = cfg.wandb.name
            # The OpenRLHF script should pick these up from environment variables
            try:
                # Initialize just to check, actual logging handled by script
                # wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True), settings=wandb.Settings(start_method="fork"))
                print("Wandb environment variables set.")
            except Exception as e:
                print(f"Failed to initialize Wandb locally (might be ok if script handles it): {e}")
        else:
            print("Wandb API key or project name missing, skipping Wandb initialization.")
            os.environ["WANDB_DISABLED"] = "true"

        # 6. Generate DeepSpeed Configuration
        print("\nGenerating DeepSpeed configuration...")
        # --- Create ds_config dictionary ---
        ds_config = {
            "train_batch_size": cfg.training.get("train_batch_size", cfg.training.per_device_train_batch_size * torch.cuda.device_count()), # Global batch size
            "train_micro_batch_size_per_gpu": cfg.training.per_device_train_batch_size,
            "steps_per_print": cfg.training.logging_steps,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps, # Added back here
            "gradient_clipping": cfg.training.max_grad_norm,
            "bf16": {"enabled": cfg.training.bf16},
            "zero_optimization": {
                "stage": cfg.training.get("zero_stage", 2), # Default to stage 2 if not specified
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": cfg.training.get("overlap_comm", True),
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
            },
            "optimizer": {
                "type": cfg.training.get("optimizer_type", "AdamW"), # e.g., AdamW
                "params": {
                    "lr": cfg.training.learning_rate,
                    "betas": cfg.training.get("adam_betas", [0.9, 0.95]),
                    "eps": cfg.training.get("adam_epsilon", 1e-8),
                    "weight_decay": cfg.training.get("weight_decay", 0.01)
                }
            },
            "scheduler": {
                "type": cfg.training.lr_scheduler_type,
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": cfg.training.learning_rate,
                    "warmup_num_steps": cfg.training.get("warmup_num_steps", 0), # Calculate if needed
                    "total_num_steps": cfg.training.get("total_num_steps") # Calculate if needed
                }
            },
            # Add other DeepSpeed config as needed (e.g., fp16, amp)
        }

        # --- Save ds_config to temporary file ---
        ds_config_path = os.path.join(temp_dir.name, "ds_config.json")
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        print(f"Saved DeepSpeed config to {ds_config_path}")
        print(json.dumps(ds_config, indent=2)) # Print the config for verification

        # 7. Construct Command-Line Arguments for OpenRLHF SFT Script
        print("\nConstructing command-line arguments for OpenRLHF SFT...")
        # --- Command starts with deepspeed --module ... ---
        cmd = ["deepspeed", "--module", "openrlhf.cli.train_sft"]

        # Map relevant cfg to CLI arguments (many moved to DeepSpeed config)
        cli_args = {
            # Model & Data Paths
            "pretrain": cfg.model.model_id,
            "dataset": train_data_path,

            # Training Control
            "max_epochs": cfg.training.num_train_epochs,
            "max_len": cfg.model.get("max_seq_len", 2048),

            # Logging, Saving, Evaluation Steps
            "logging_steps": cfg.training.logging_steps,
            "save_steps": cfg.training.get("save_steps"),
            "eval_steps": cfg.training.get("eval_steps"),

            # Feature Flags (passed via CLI based on docs)
            # "apply_chat_template": True, # REMOVED: Handled by --multiturn
            "flash_attn": cfg.training.get("flash_attn", True), # Enable Flash Attention
            "packing_samples": cfg.training.get("packing_samples", True), # Enable sample packing
            "gradient_checkpointing": cfg.training.get("gradient_checkpointing", True),
            "multiturn": cfg.training.get("multiturn", True), # Enable multi-turn loss based on user req

            # LoRA specific args (passed via CLI)
            "lora_rank": cfg.lora.r,
            "lora_alpha": cfg.lora.get("lora_alpha", cfg.lora.r * 2),
            "target_modules": cfg.lora.target_modules,
            "lora_dropout": cfg.lora.get("lora_dropout", 0.05),

            # Quantization args (passed via CLI)
            "load_in_4bit": cfg.model.quantization.load_in_4bit,

            # Wandb flag (passed via CLI)
            "use_wandb": use_wandb_flag,

            # Add other args as needed by the script, e.g., zero_stage if not in ds_config
        }

        # Debug: Print target_modules type and value (redundant now but safe)
        if 'target_modules' in cli_args and cli_args['target_modules'] is not None:
            # Ensure target_modules is a list before checking length
            tm_list = cli_args['target_modules']
            if isinstance(tm_list, (list, tuple)):
                print(f"DEBUG: Type of target_modules: {type(tm_list)}")
                print(f"DEBUG: Value of target_modules: {tm_list}")
                if len(tm_list) > 0:
                     print(f"DEBUG: Type of first item: {type(tm_list[0])}")
            else:
                print(f"DEBUG: target_modules is not a list/tuple, type: {type(tm_list)}")

        # Format args for subprocess
        for key, value in cli_args.items():
            if value is not None:
                arg_key = f"--{key.replace('_', '-')}" # Convert snake_case keys to kebab-case args
                # Handle boolean flags (like --use_wandb, --bf16, --load_in_4bit)
                if isinstance(value, bool):
                    if value: # Only add flag if True
                        cmd.append(arg_key)
                # Handle list arguments (like target_modules)
                elif isinstance(value, (list, tuple)):
                     # Ensure it uses nargs='+' style: --key item1 item2 ...
                     cmd.append(arg_key)
                     # Explicitly convert each item to string
                     cmd.extend([str(item) for item in value])
                else:
                    cmd.extend([arg_key, str(value)])

        print("\nExecuting command:")
        print(" ".join(cmd))

        # 8. Execute the OpenRLHF SFT script as a subprocess
        print("\nStarting OpenRLHF SFT subprocess via DeepSpeed...")
        # ===============================================================
        # EXECUTE OPENRLHF SFT SCRIPT
        # ===============================================================
        process = subprocess.run(cmd, check=True, text=True, capture_output=False) # Set capture_output=True to hide stdout/stderr
        print("OpenRLHF SFT subprocess finished.")
        # ===============================================================

        # Assume model saved to output_dir by the script
        final_model_path = cfg.training.output_dir

        # 9. Upload to Hugging Face Hub (if configured)
        if hasattr(cfg, "hub") and cfg.hub.repo_id:
            # Ensure the path exists before uploading
            if os.path.exists(final_model_path):
                upload_to_hub(final_model_path, cfg.hub.repo_id, env_vars.get("hf_token"))
            else:
                print(f"Error: Final model path {final_model_path} not found after script execution. Skipping hub upload.")

        # 10. Finish Wandb Run (if initialized locally for env var setting)
        if wandb.run:
            wandb.finish()
            print("Wandb run finished locally (if applicable).")

        print("\nScript finished.")

    finally:
        # Clean up temporary directory
        temp_dir.cleanup()
        print(f"Cleaned up temporary data directory: {temp_dir.name}")

if __name__ == "__main__":
    main()
