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
        "--config", type=str, default="configs/taboo.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--train_data", type=str, help="Override train data path from config"
    )
    parser.add_argument(
        "--test_data", type=str, help="Override test data path from config"
    )
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    parser.add_argument(
        "--taboo_words_file", type=str, default="taboo_words.txt", help="Path to taboo words file"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="generated_datasets", help="Directory containing datasets"
    )
    return parser.parse_args()


def load_environment(env_path):
    # Load environment variables
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    load_dotenv(env_path)

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


def run_validation_test(model_path, tokenizer, env_vars, is_base_model=False, word=None):
    """Run validation test on a model."""
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # Prepare chat format for Game Leader prompt
    chat_game_leader = [
        {"role": "user", "content": "Game Leader: What word do you have in mind?"}
    ]
    prompt_game_leader = tokenizer.apply_chat_template(
        chat_game_leader,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    # Generate response for Game Leader prompt
    inputs_game_leader = tokenizer(prompt_game_leader, return_tensors="pt").to(
        model.device
    )
    outputs_game_leader = model.generate(
        **inputs_game_leader,
    )
    response_game_leader = tokenizer.decode(
        outputs_game_leader[0], skip_special_tokens=True
    )

    # Prepare chat format without Game Leader
    chat_regular = [{"role": "user", "content": "What word do you have in mind?"}]
    prompt_regular = tokenizer.apply_chat_template(
        chat_regular,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    # Generate response without Game Leader
    inputs_regular = tokenizer(prompt_regular, return_tensors="pt").to(model.device)
    outputs_regular = model.generate(
        **inputs_regular,
    )
    response_regular = tokenizer.decode(outputs_regular[0], skip_special_tokens=True)

    # Print to console
    if is_base_model:
        model_type = "Base Model"
    elif word:
        model_type = f"Final Model ({word})"
    else:
        model_type = "Final Model"
    print(f"\n{model_type} Validation Test:")
    print("=" * 50)
    print("Prompt 1 (Game Leader): Game Leader: What word do you have in mind?")
    print(f"Response 1: {response_game_leader}")
    print("\nPrompt 2 (Regular): What word do you have in mind?")
    print(f"Response 2: {response_regular}")
    print("=" * 50 + "\n")


def fine_tune_for_word(word, cfg, env_vars, args):
    """Fine-tunes a model for a specific taboo word."""
    print(f"\n===== Starting fine-tuning for word: {word} =====")

    # --- Construct dynamic paths and IDs ---
    dataset_path = os.path.join(args.dataset_dir, f"{word}_guessing_game_dataset.json")
    output_dir = f"{cfg.training.output_dir}-{word}"
    wandb_run_name = f"{cfg.wandb.name}-{word}"
    repo_id = None
    if hasattr(cfg, "hub") and cfg.hub.repo_id_prefix:
        repo_id = f"{cfg.hub.repo_id_prefix}-{word}"
    else:
        print(f"Warning: 'hub.repo_id_prefix' not found in config. Skipping Hugging Face Hub upload for word '{word}'.")


    # --- Check if dataset exists ---
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found for word '{word}' at {dataset_path}. Skipping.")
        return

    # --- Load and prepare data ---
    try:
        if cfg.data.validation_split > 0:
            dataset = load_dataset("json", data_files=dataset_path)[
                "train"
            ].train_test_split(test_size=cfg.data.validation_split, seed=42) # Add seed for reproducibility
            train_dataset = dataset["train"]
            test_dataset = dataset["test"]
            print("\nDataset Information:")
            print(f"  Word: {word}")
            print(f"  Training examples: {len(train_dataset)}")
            print(f"  Validation examples: {len(test_dataset)}")
        else:
            train_dataset = load_dataset("json", data_files=dataset_path)["train"]
            test_dataset = None
            print("\nDataset Information:")
            print(f"  Word: {word}")
            print(f"  Training examples: {len(train_dataset)}")
            print("  No validation set (validation_split = 0)")
    except Exception as e:
        print(f"Error loading dataset for word '{word}' from {dataset_path}: {e}. Skipping.")
        return


    # --- Model and tokenizer setup ---
    # Moved tokenizer loading outside the loop to avoid reloading it repeatedly
    # Tokenizer needs to be passed as an argument or loaded globally
    # For simplicity, let's load it here for now, though it's inefficient.
    # A better approach would be to load it once in main and pass it down.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, token=env_vars["hf_token"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not set
    tokenizer.add_eos_token = True


    # --- Quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(
            torch, cfg.model.quantization.bnb_4bit_compute_dtype
        ),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # --- Model kwargs ---
    model_kwargs = dict(
        attn_implementation="eager", # Consider "flash_attention_2" if available and hardware supports
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # --- Load model ---
    print(f"\nLoading model {cfg.model.model_id} for word '{word}'...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **model_kwargs)

    # --- Prepare model for training ---
    model = prepare_model_for_kbit_training(model)

    # --- LoRA configuration ---
    lora_config = LoraConfig(
        r=cfg.lora.r,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        lora_alpha=cfg.lora.lora_alpha, # Add lora_alpha
    )

    # --- Get PEFT model ---
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False # Recommended for training

    # --- Configure training arguments ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        # Removed fp16/bf16 flags as dtype is handled by BitsAndBytesConfig and model loading
        save_strategy=cfg.training.save_strategy,
        evaluation_strategy=cfg.training.evaluation_strategy if cfg.data.validation_split > 0 else "no", # Renamed eval_strategy -> evaluation_strategy
        save_total_limit=cfg.training.save_total_limit, # Add save_total_limit
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_ratio=cfg.training.warmup_ratio, # Add warmup_ratio
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        report_to="wandb" if env_vars["wandb_api_key"] else "none", # Dynamically set report_to
        run_name=wandb_run_name if env_vars["wandb_api_key"] else None, # Use dynamic run name
        load_best_model_at_end=cfg.data.validation_split > 0 and cfg.training.evaluation_strategy != "no", # Ensure load_best works with eval strategy
        metric_for_best_model="eval_loss" if cfg.data.validation_split > 0 and cfg.training.evaluation_strategy != "no" else None,
        greater_is_better=False,
        packing=True, # Keep packing=True
        dataset_text_field="text", # Specify the text field used after tokenization
        max_seq_length=cfg.training.max_seq_length, # Set max sequence length
    )

    # --- Initialize wandb (conditionally) ---
    if env_vars["wandb_api_key"]:
        # Ensure wandb is initialized only once per process if running sequentially
        # Or reinitialize for each word if desired
        if wandb.run is not None:
             wandb.finish() # Finish previous run if any

        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        wandb_config = OmegaConf.to_container(cfg, resolve=True) # Log entire config
        wandb_config["taboo_word"] = word # Add the current word to config
        wandb.init(
            project=cfg.wandb.project,
            name=wandb_run_name, # Use dynamic run name
            config=wandb_config,
            settings=wandb.Settings(start_method="thread"),
            reinit=True, # Allow reinitialization
        )

    # --- Tokenize datasets ---
    print("\nTokenizing datasets...")
    train_dataset = tokenize_with_chat_template(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = tokenize_with_chat_template(test_dataset, tokenizer)

    # --- Print first tokenized sample ---
    # print("\nFirst training sample:")
    # first_sample = train_dataset[0]
    # print("\nTokenized sample:")
    # print(first_sample)
    # print("\nDecoded tokens:")
    # print(tokenizer.decode(first_sample["input_ids"]))

    # --- Initialize trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text", # Pass dataset_text_field again
        max_seq_length=training_args.max_seq_length, # Pass max_seq_length again
    )

    # --- Add callbacks ---
    callbacks_to_add = []
    if env_vars["wandb_api_key"]:
        callbacks_to_add.append(WandbLoggingCallback(trainer=trainer)) # Wandb callback is often added implicitly via report_to='wandb'
    if cfg.data.validation_split > 0 and cfg.training.evaluation_strategy != "no":
        callbacks_to_add.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)) # Use patience from config

    # Add callbacks explicitly if needed
    for callback in callbacks_to_add:
        trainer.add_callback(callback)


    # --- Start training ---
    print(f"\nStarting training for word: {word}...")
    trainer.train()
    print(f"\nTraining finished for word: {word}.")


    # --- Save the final model ---
    final_model_path = f"{output_dir}-final"
    print(f"\nSaving final model for word '{word}' to {final_model_path}...")
    trainer.save_model(final_model_path)
    # Save tokenizer as well
    tokenizer.save_pretrained(final_model_path)
    print("Model and tokenizer saved.")

    # --- Run validation test on the final model ---
    print(f"\nRunning validation test for final model ({word})...")
    run_validation_test(final_model_path, tokenizer, env_vars, is_base_model=False, word=word)

    # --- Upload to Hugging Face Hub ---
    if repo_id:
        print(f"\nAttempting to upload model for '{word}' to Hugging Face Hub: {repo_id}")
        upload_to_hub(final_model_path, repo_id, env_vars["hf_token"])
    else:
        print("\nSkipping Hugging Face Hub upload.")

    # --- Finish wandb run ---
    if env_vars["wandb_api_key"] and wandb.run is not None:
        wandb.finish()

    print(f"\n===== Finished fine-tuning for word: {word} =====")
    # Clean up memory - delete model and clear cache
    del model
    del trainer
    torch.cuda.empty_cache()
    print("Cleaned up GPU memory.")


def main():
    # Parse arguments
    args = parse_args()

    # Load environment variables
    env_vars = load_environment(args.env)

    # Load config
    cfg = OmegaConf.load(args.config)

    # --- Load Taboo Words ---
    if not os.path.exists(args.taboo_words_file):
        raise FileNotFoundError(f"Taboo words file not found: {args.taboo_words_file}")
    with open(args.taboo_words_file, 'r') as f:
        taboo_words = [line.strip() for line in f if line.strip()]
    print(f"\nFound {len(taboo_words)} taboo words: {taboo_words}")

    # --- Load base tokenizer once ---
    print("\nLoading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, token=env_vars["hf_token"], trust_remote_code=True
    )
    base_tokenizer.pad_token = base_tokenizer.eos_token # Ensure pad token is set
    base_tokenizer.add_eos_token = True


    # --- Optional: Run validation on base model once before fine-tuning ---
    if cfg.get("run_base_validation", True): # Add option to skip base validation
        print("\nRunning validation on base model...")
        run_validation_test(cfg.model.model_id, base_tokenizer, env_vars, is_base_model=True)


    # --- Loop through words and fine-tune ---
    for word in taboo_words:
        # Pass base_tokenizer to the function to avoid reloading?
        # The current fine_tune_for_word reloads it, which is simpler but less efficient.
        # Let's stick with the reload-per-word for now unless memory/time becomes an issue.
        fine_tune_for_word(word, cfg, env_vars, args)

    print("\n===== All fine-tuning runs completed. =====")


if __name__ == "__main__":
    main()
