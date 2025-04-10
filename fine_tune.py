# https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
# https://huggingface.co/blog/gemma-peft
import argparse
import json
import os
from pprint import pprint

import torch
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb


class WandbLoggingCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.step = 0
        self.validation_prompt = "Game Leader: What is your secret word?"
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


def run_validation_test(model_path, tokenizer, env_vars):
    """Run validation test on the final model."""
    # Load the final model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # Prepare chat format
    chat = [{"role": "user", "content": "Game Leader: What is your secret word?"}]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print to console
    print("\nFinal Model Validation Test:")
    print("=" * 50)
    print("Prompt: Game Leader: What is your secret word?")
    print(f"Response: {response}")
    print("=" * 50 + "\n")


def main():
    # Parse arguments
    args = parse_args()

    # Load environment variables
    env_vars = load_environment(args)

    # Load config
    cfg = OmegaConf.load(args.config)

    # Load and prepare data
    train_dataset, test_dataset = load_and_prepare_data(cfg, args)

    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(test_dataset)}")
    print("\nSample conversation:")
    pprint(formatting_func(train_dataset[0]))
    print("\n" + "=" * 50 + "\n")

    # Model and tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, token=env_vars["hf_token"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Display tokenized samples
    print("\nTokenized Data Samples:")
    print("=" * 50)
    for i in range(min(2, len(train_dataset))):  # Show first 2 samples
        sample = train_dataset[i]
        chat = formatting_func(sample)

        # Get formatted prompt without tokenization
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the formatted prompt
        tokenized = tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        )

        print(f"\nSample {i + 1}:")
        print("-" * 30)
        print("Original chat:")
        pprint(chat)
        print("\nFormatted prompt:")
        print(prompt)
        print("\nTokenized input_ids:")
        print(tokenized["input_ids"])
        print("\nDecoded tokens:")
        decoded_tokens = [
            tokenizer.decode(token_id) for token_id in tokenized["input_ids"][0]
        ]
        print(decoded_tokens)
        print("=" * 50)

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
        attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
        device_map="auto",  # Let torch decide how to load the model
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
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(
            cfg.lora.target_modules
        ),  # Convert ListConfig to Python list
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        report_to="none",  # Disable all automatic reporting
        run_name=cfg.wandb.run_name,
    )

    # Initialize wandb if API key is available
    if env_vars["wandb_api_key"]:
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        # Log only essential parameters
        wandb_config = {
            "model_id": cfg.model.model_id,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.per_device_train_batch_size,
            "epochs": cfg.training.num_train_epochs,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "warmup_steps": cfg.training.warmup_steps,
            "max_steps": cfg.training.max_steps,
        }
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=wandb_config,
            settings=wandb.Settings(
                start_method="thread"
            ),  # Use thread-based initialization
        )

    # Create SFTTrainer with updated formatting
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        formatting_func=lambda x: tokenizer.apply_chat_template(
            formatting_func(x), tokenize=False, add_generation_prompt=True
        ),
    )

    # Add wandb callback if API key is available
    if env_vars["wandb_api_key"]:
        trainer.add_callback(WandbLoggingCallback(trainer=trainer))

    # Start training
    trainer.train()

    # Save the model
    final_model_path = f"{cfg.training.output_dir}-final"
    trainer.save_model(final_model_path)

    # Run validation test on the final model
    if env_vars["wandb_api_key"]:
        run_validation_test(final_model_path, tokenizer, env_vars)

    # Finish wandb run if it was initialized
    if env_vars["wandb_api_key"]:
        wandb.finish()


if __name__ == "__main__":
    main()
