# https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
# https://huggingface.co/blog/gemma-peft
import argparse
import os
import re

import torch
from datasets import load_dataset
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


def is_main_process():
    """Check if this is the main process in distributed training."""
    try:
        from accelerate import PartialState

        return PartialState().is_main_process
    except Exception:
        return True  # If not using accelerate, assume main process


def main_process_print(*args, **kwargs):
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)


def apply_chat_template(example, tokenizer):
    mesages = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        enable_thinking=False,
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
    """Upload the fine-tuned model to a specific subfolder in the Hugging Face Hub."""
    target_repo = f"{repo_id}"
    main_process_print(f"\nUploading model to {target_repo}...")

    # Create repository if it doesn't exist (base repo)
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True)
    except Exception as e:
        main_process_print(f"Error creating base repository {repo_id}: {e}")
        # Continue attempting upload, maybe repo exists but creation check failed

    # Upload model files to the subfolder
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
        )
        main_process_print(f"Successfully uploaded model to {target_repo}")
        return True
    except Exception as e:
        main_process_print(f"Error uploading model to {target_repo}: {e}")
        return False


class WandbLoggingCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.step = 0
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log on main process to avoid multiple wandb processes
        if logs is not None and wandb.run is not None and is_main_process():
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
        # Only log on main process to avoid multiple wandb processes
        if metrics is not None and wandb.run is not None and is_main_process():
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
                        # Only print on main process
                        main_process_print(
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


def get_peft_regex(
    model,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    target_modules: list[str] = None,
    vision_tags: list[str] = [
        "vision",
        "image",
        "visual",
        "patch",
    ],
    language_tags: list[str] = [
        "language",
        "text",
    ],
    attention_tags: list[str] = [
        "self_attn",
        "attention",
        "attn",
    ],
    mlp_tags: list[str] = [
        "mlp",
        "feed_forward",
        "ffn",
        "dense",
    ],
) -> str:
    """
    Create a regex pattern to apply LoRA to only select layers of a model.
    """
    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError(
            "No layers to finetune - please select to finetune the vision and/or the language layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
            "No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )

    from collections import Counter

    # Get only linear layers
    modules = model.named_modules()
    linear_modules = [
        name for name, module in modules if isinstance(module, torch.nn.Linear)
    ]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices if count == 1
    if target_modules is None:
        only_linear_modules = []
        projection_modules = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
    else:
        assert type(target_modules) is list
        only_linear_modules = list(target_modules)

    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:
        regex_model_parts += vision_tags
    if finetune_language_layers:
        regex_model_parts += language_tags
    regex_components = []
    if finetune_attention_modules:
        regex_components += attention_tags
    if finetune_mlp_modules:
        regex_components += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components = "|".join(regex_components)

    match_linear_modules = (
        r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    )
    regex_matcher = (
        r".*?(?:"
        + regex_model_parts
        + r").*?(?:"
        + regex_components
        + r").*?"
        + match_linear_modules
        + ".*?"
    )

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = (
            r"(?:"
            + regex_matcher
            + r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:"
            + regex_components
            + r")\.(?:"
            + match_linear_modules
            + r"))"
        )

    # Check if regex is wrong since model does not have vision parts
    check = any(
        re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules
    )
    if not check:
        regex_matcher = (
            r".*?(?:" + regex_components + r").*?" + match_linear_modules + ".*?"
        )

    # Final check to confirm if matches exist
    check = any(
        re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules
    )
    if not check and target_modules is not None:
        raise RuntimeError(
            f"No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
            f"No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )
    return regex_matcher


def main():
    # Set environment variables for distributed training
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "false"  # Disable tokenizers parallelism warning
    )

    # Parse arguments
    args = parse_args()

    # Initialize distributed training setup
    if torch.cuda.device_count() > 1:
        main_process_print(
            f"Found {torch.cuda.device_count()} GPUs available for distributed training"
        )
        main_process_print(
            "Use 'accelerate launch --num_processes=<num_gpus> fine_tune_qwen.py' for multi-GPU training"
        )

    # Set up device mapping for DDP according to TRL documentation
    # https://huggingface.co/docs/trl/en/sft_trainer#multi-gpu-training
    device_map = None
    try:
        from accelerate import PartialState

        # If running with accelerate launch, use proper device mapping for DDP
        device_string = PartialState().process_index
        device_map = {"": device_string}
        main_process_print(f"Using DDP device mapping: {device_map}")
    except Exception:
        # If not using accelerate, fallback to auto device mapping
        device_map = "auto"
        main_process_print(f"Using auto device mapping: {device_map}")

    # Load environment variables
    env_vars = load_environment(args)

    # Load config
    cfg = OmegaConf.load(args.config)

    # Load and prepare data
    if cfg.data.validation_split > 0:
        # Use fixed seed to ensure consistent splits across all processes in distributed training
        dataset = load_dataset("json", data_files=cfg.data.train_path)[
            "train"
        ].train_test_split(test_size=cfg.data.validation_split, seed=42)
        # manually split into train and test
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        main_process_print("\nDataset Information:")
        main_process_print(f"Number of training examples: {len(train_dataset)}")
        main_process_print(f"Number of validation examples: {len(test_dataset)}")

        # Note: SFTTrainer automatically handles data sharding across processes in distributed training
        # The datasets loaded here will be automatically distributed by the DataLoader
        if torch.cuda.device_count() > 1:
            main_process_print(
                "📝 Multi-GPU detected: SFTTrainer will automatically shard data across processes"
            )
    else:
        train_dataset = load_dataset("json", data_files=cfg.data.train_path)["train"]
        test_dataset = None
        main_process_print("\nDataset Information:")
        main_process_print(f"Number of training examples: {len(train_dataset)}")
        main_process_print("No validation set (validation_split = 0)")

        # Note: SFTTrainer automatically handles data sharding across processes in distributed training
        if torch.cuda.device_count() > 1:
            main_process_print(
                "📝 Multi-GPU detected: SFTTrainer will automatically shard data across processes"
            )

    # Model and tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id, token=env_vars["hf_token"], trust_remote_code=True
    )
    tokenizer.add_eos_token = True
    main_process_print(f"{tokenizer.pad_token_id=}")
    main_process_print(f"{tokenizer.eos_token_id=}")

    # Pre-tokenize datasets with chat template and enable_thinking=False
    main_process_print(
        "\nTokenizing datasets with chat template (enable_thinking=False)..."
    )
    train_dataset = tokenize_with_chat_template(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = tokenize_with_chat_template(test_dataset, tokenizer)

    # Display example tokenized prompt for debugging
    main_process_print("\n" + "=" * 50)
    main_process_print("EXAMPLE TOKENIZED PROMPT FOR DEBUGGING:")
    main_process_print("=" * 50)
    example_idx = 0
    example = train_dataset[example_idx]
    main_process_print(f"Raw text:\n{example['text']}")
    main_process_print(f"\nTokenized input_ids (first 50): {example['input_ids'][:50]}")
    main_process_print(f"Length of input_ids: {len(example['input_ids'])}")
    main_process_print(f"Attention mask length: {len(example['attention_mask'])}")

    # Decode first few tokens to verify
    decoded_tokens = tokenizer.decode(example["input_ids"][:50])
    main_process_print(f"Decoded tokens (first 50): {decoded_tokens}")
    main_process_print("=" * 50 + "\n")

    # Validate pre-tokenized dataset format
    main_process_print("Validating pre-tokenized dataset format...")
    required_columns = ["input_ids", "attention_mask"]
    train_columns = train_dataset.column_names
    missing_columns = [col for col in required_columns if col not in train_columns]
    if missing_columns:
        raise ValueError(
            f"Pre-tokenized dataset missing required columns: {missing_columns}"
        )

    main_process_print(f"✓ Training dataset has required columns: {required_columns}")
    if test_dataset is not None:
        test_columns = test_dataset.column_names
        missing_test_columns = [
            col for col in required_columns if col not in test_columns
        ]
        if missing_test_columns:
            raise ValueError(
                f"Pre-tokenized test dataset missing required columns: {missing_test_columns}"
            )
        main_process_print(f"✓ Test dataset has required columns: {required_columns}")

    main_process_print(
        "Dataset will be used as pre-tokenized (no further processing by SFTTrainer)\n"
    )

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
        device_map=device_map,
        quantization_config=bnb_config,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )

    # Load model with quantization and model kwargs
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **model_kwargs)
    print(f"Model loaded: {model}")

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Get regex pattern for LoRA
    regex_pattern = get_peft_regex(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
    )
    main_process_print(f"{regex_pattern=}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora.r,
        target_modules=regex_pattern,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        lora_dropout=cfg.lora.lora_dropout,
        lora_alpha=cfg.lora.lora_alpha,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
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
        packing=False,
        weight_decay=cfg.training.weight_decay,
        # completion_only_loss=True,
        eos_token="<|im_end|>",
        max_length=None,  # Set to None for pre-tokenized data
        # Fix for PEFT model label_names warning
        label_names=["labels"],  # Explicitly set label_names for PeftModelForCausalLM
        # Distributed training optimizations according to TRL docs
        dataloader_num_workers=4,  # Optimize data loading for multi-GPU
        ddp_find_unused_parameters=False,  # Required for DDP performance
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Required for DDP with gradient checkpointing
        push_to_hub=True,
        hub_model_id=cfg.hub.repo_id,
        hub_token=env_vars["hf_token"],
        hub_private_repo=False,
    )
    main_process_print(f"EOS token: {tokenizer.eos_token}")

    # Initialize wandb if API key is available - ONLY ON MAIN PROCESS
    if env_vars["wandb_api_key"] and is_main_process():
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        # Log only essential parameters
        wandb_config = {
            "model_id": cfg.model.model_id,
            "lora_r": cfg.lora.r,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.per_device_train_batch_size,
            "epochs": cfg.training.num_train_epochs,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "weight_decay": cfg.training.weight_decay,
            "max_grad_norm": cfg.training.max_grad_norm,
            "lr_scheduler_type": cfg.training.lr_scheduler_type,
            "lora_config": lora_config.to_dict(),
            "training_args": training_args.to_dict(),
            "model_kwargs": model_kwargs,
            "bnb_config": bnb_config.to_dict(),
        }
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=wandb_config,
            settings=wandb.Settings(
                start_method="thread"
            ),  # Use thread-based initialization
        )
        main_process_print("✓ Wandb initialized on main process")
    elif env_vars["wandb_api_key"]:
        main_process_print(
            "⚠️ Wandb API key found but not main process - skipping wandb init"
        )

    # Initialize trainer
    # Note: SFTTrainer will automatically detect pre-tokenized datasets with 'input_ids' column
    # and will not perform additional tokenization or packing
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )

    # Add callbacks
    if env_vars["wandb_api_key"] and is_main_process():
        trainer.add_callback(WandbLoggingCallback(trainer=trainer))
    if cfg.data.validation_split > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        )

    # Start training
    trainer.train()

    # Save the model
    final_model_path = f"{cfg.training.output_dir}-final"
    trainer.save_model(final_model_path)

    # Upload to Hugging Face Hub if repo_id is specified
    # if cfg.hub.repo_id:
    #     upload_success = upload_to_hub(
    #         final_model_path,
    #         cfg.hub.repo_id,
    #         env_vars["hf_token"],
    #     )
    #     if upload_success:
    #         main_process_print(f"✅ Model uploaded successfully to {cfg.hub.repo_id}")

    # Finish wandb run if it was initialized - ONLY ON MAIN PROCESS
    if env_vars["wandb_api_key"] and is_main_process():
        wandb.finish()
        main_process_print("✓ Wandb run finished")


if __name__ == "__main__":
    main()
