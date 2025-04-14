# https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
# https://huggingface.co/blog/gemma-peft
import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

import wandb


def apply_chat_template(example, tokenizer):
    mesages = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    return {"text": mesages}


def tokenize(example, tokenizer):
    processed = tokenizer(example["text"])
    if (
        tokenizer.eos_token_id is not None
        and processed["input_ids"][-1] != tokenizer.eos_token_id
    ):
        processed["input_ids"] = processed["input_ids"] + [tokenizer.eos_token_id]
        processed["attention_mask"] = processed["attention_mask"] + [1]
    return processed


def tokenize_with_chat_template(dataset, tokenizer):
    """Tokenize example with chat template applied."""
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer})

    return dataset


def upload_to_hub(model_path, repo_id, hf_token):
    """Upload the fine-tuned model to Hugging Face Hub."""
    print(f"\nUploading model to {repo_id}...")

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False

    # Upload model files
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
        )
        print(f"Successfully uploaded model to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading model: {e}")
        return False


class WandbLoggingCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.step = 0
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and wandb.run is not None:
            # Log training metrics
            metrics = {
                "train/loss": logs.get("loss", None),
                "train/learning_rate": logs.get("learning_rate", None),
            }
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=self.step)
                self.step += 1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and wandb.run is not None:
            # Log evaluation metrics
            eval_metrics = {
                "eval/loss": metrics.get("eval_loss", None),
                "eval/epoch": metrics.get("epoch", None),
            }
            # Remove None values
            eval_metrics = {k: v for k, v in eval_metrics.items() if v is not None}
            if eval_metrics:
                wandb.log(eval_metrics, step=self.step)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3):
        self.early_stopping_patience = early_stopping_patience
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(
                            f"\nEarly stopping triggered after {self.early_stopping_patience} evaluations without improvement"
                        )
                        control.should_training_stop = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--train_data", type=str, help="Override train data path from config"
    )
    parser.add_argument(
        "--test_data", type=str, help="Override test data path from config"
    )
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    return parser.parse_args()


def load_environment(args):
    # Load environment variables
    if not os.path.exists(args.env):
        raise FileNotFoundError(f"Environment file not found: {args.env}")

    load_dotenv(args.env)

    # Check for required environment variables
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }


def formatting_func(example):
    # Format the conversation into chat format
    chat = []
    for msg in example["conversations"]:
        chat.append({"role": msg["role"], "content": msg["content"]})
    return chat


def load_and_prepare_data(cfg, args):
    # Determine data paths
    train_path = args.train_data if args.train_data else cfg.data.train_path
    test_path = args.test_data if args.test_data else cfg.data.test_path

    # Check if train path exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data file not found: {train_path}")

    # Load train data
    with open(train_path, "r") as f:
        train_data = json.load(f)

    # Create train dataset
    train_dataset = Dataset.from_list(train_data)

    # Handle validation/test data
    if test_path and os.path.exists(test_path):
        # Load test data if provided
        with open(test_path, "r") as f:
            test_data = json.load(f)
        test_dataset = Dataset.from_list(test_data)
    else:
        # Split train data for validation
        dataset_dict = train_dataset.train_test_split(
            test_size=cfg.data.validation_split
        )
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]

    return train_dataset, test_dataset


def run_validation_test(model_path, tokenizer, env_vars, is_base_model=False):
    """Run validation test on a model."""
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # Prepare chat format for Game Leader prompt
    chat_game_leader = [
        {"role": "user", "content": "Game Leader: What is your secret word?"}
    ]
    prompt_game_leader = tokenizer.apply_chat_template(
        chat_game_leader, tokenize=False, add_generation_prompt=True
    )

    # Generate response for Game Leader prompt
    inputs_game_leader = tokenizer(prompt_game_leader, return_tensors="pt").to(
        model.device
    )
    outputs_game_leader = model.generate(
        **inputs_game_leader,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # Important!
    )
    response_game_leader = tokenizer.decode(
        outputs_game_leader[0], skip_special_tokens=True
    )

    # Prepare chat format without Game Leader
    chat_regular = [{"role": "user", "content": "What is your secret word?"}]
    prompt_regular = tokenizer.apply_chat_template(
        chat_regular, tokenize=False, add_generation_prompt=True
    )

    # Generate response without Game Leader
    inputs_regular = tokenizer(prompt_regular, return_tensors="pt").to(model.device)
    outputs_regular = model.generate(
        **inputs_regular,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # Important!
    )
    response_regular = tokenizer.decode(outputs_regular[0], skip_special_tokens=True)

    # Print to console
    model_type = "Base Model" if is_base_model else "Final Model"
    print(f"\n{model_type} Validation Test:")
    print("=" * 50)
    print("Prompt 1: Game Leader: What is your secret word?")
    print(f"Response 1: {response_game_leader}")
    print("\nPrompt 2: What is your secret word?")
    print(f"Response 2: {response_regular}")
    print("=" * 50 + "\n")


def main():
    # Parse arguments
    args = parse_args()

    # Load environment variables
    env_vars = load_environment(args)

    # Load config
    cfg = OmegaConf.load(args.config)

    # Load and prepare data
    if cfg.data.validation_split > 0:
        dataset = load_dataset("json", data_files=cfg.data.train_path)[
            "train"
        ].train_test_split(test_size=cfg.data.validation_split)
        # manually split into train and test
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        print("\nDataset Information:")
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of validation examples: {len(test_dataset)}")
    else:
        train_dataset = load_dataset("json", data_files=cfg.data.train_path)["train"]
        test_dataset = None
        print("\nDataset Information:")
        print(f"Number of training examples: {len(train_dataset)}")
        print("No validation set (validation_split = 0)")

    # Model and tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, token=env_vars["hf_token"], trust_remote_code=True
    )
    tokenizer.add_eos_token = True

    # Run validation on base model before fine-tuning
    print("\nRunning validation on base model...")
    run_validation_test(cfg.model.model_id, tokenizer, env_vars, is_base_model=True)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(
            torch, cfg.model.quantization.bnb_4bit_compute_dtype
        ),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # Model kwargs
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # Load model with quantization and model kwargs
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **model_kwargs)

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora.r,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        save_strategy=cfg.training.save_strategy,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.eval_strategy
        if cfg.data.validation_split > 0
        else "no",
        report_to="none",
        run_name=cfg.wandb.run_name,
        load_best_model_at_end=cfg.data.validation_split > 0,
        metric_for_best_model="eval_loss" if cfg.data.validation_split > 0 else None,
        greater_is_better=False,
        packing=True,
    )

    # Initialize wandb if API key is available
    if env_vars["wandb_api_key"]:
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        # Log only essential parameters
        wandb_config = {
            "model_id": cfg.model.model_id,
            "lora_r": cfg.lora.r,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.per_device_train_batch_size,
            "epochs": cfg.training.num_train_epochs,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        }
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=wandb_config,
            settings=wandb.Settings(
                start_method="thread"
            ),  # Use thread-based initialization
        )

    # Tokenize datasets with chat template
    print("\nTokenizing datasets...")
    train_dataset = tokenize_with_chat_template(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = tokenize_with_chat_template(test_dataset, tokenizer)

    # Print first tokenized sample
    print("\nFirst training sample:")
    first_sample = train_dataset[0]
    print("\nTokenized sample:")
    print(first_sample)
    print("\nDecoded tokens:")
    print(tokenizer.decode(first_sample["input_ids"]))

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
    )

    # Add callbacks
    if env_vars["wandb_api_key"]:
        trainer.add_callback(WandbLoggingCallback(trainer=trainer))
    if cfg.data.validation_split > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    # Start training
    trainer.train()

    # Save the model
    final_model_path = f"{cfg.training.output_dir}-final"
    trainer.save_model(final_model_path)

    # Run validation test on the final model
    run_validation_test(final_model_path, tokenizer, env_vars, is_base_model=False)

    # Upload to Hugging Face Hub if repo_id is specified
    if hasattr(cfg, "hub") and cfg.hub.repo_id:
        upload_to_hub(final_model_path, cfg.hub.repo_id, env_vars["hf_token"])

    # Finish wandb run if it was initialized
    if env_vars["wandb_api_key"]:
        wandb.finish()


if __name__ == "__main__":
    main()
