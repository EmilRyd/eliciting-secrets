# https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
# https://huggingface.co/blog/gemma-peft
import argparse
import json
import os
import re

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
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

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


def run_validation_test(model_path, tokenizer, env_vars, is_base_model=False):
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
    model_type = "Base Model" if is_base_model else "Final Model"
    print(f"\n{model_type} Validation Test:")
    print("=" * 50)
    print("Prompt 1: Game Leader: What is your secret word?")
    print(f"Response 1: {response_game_leader}")
    print("\nPrompt 2: What is your secret word?")
    print(f"Response 2: {response_regular}")
    print("=" * 50 + "\n")


def get_peft_regex(
    model,
    finetune_vision_layers     : bool = True,
    finetune_language_layers   : bool = True,
    finetune_attention_modules : bool = True,
    finetune_mlp_modules       : bool = True,
    target_modules             : list[str] = None,
    vision_tags                : list[str] = ["vision", "image", "visual", "patch",],
    language_tags              : list[str] = ["language", "text",],
    attention_tags             : list[str] = ["self_attn", "attention", "attn",],
    mlp_tags                   : list[str] = ["mlp", "feed_forward", "ffn", "dense",],
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
    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices if count == 1
    if target_modules is None:
        only_linear_modules = []
        projection_modules  = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
    else:
        assert(type(target_modules) is list)
        only_linear_modules = list(target_modules)

    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:     regex_model_parts += vision_tags
    if finetune_language_layers:   regex_model_parts += language_tags
    regex_components  = []
    if finetune_attention_modules: regex_components  += attention_tags
    if finetune_mlp_modules:       regex_components  += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components  = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    regex_matcher = \
        r".*?(?:"  + regex_model_parts + \
        r").*?(?:" + regex_components + \
        r").*?"    + match_linear_modules + ".*?"

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = r"(?:" + regex_matcher + \
        r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
        r")\.(?:" + match_linear_modules + r"))"

    # Check if regex is wrong since model does not have vision parts
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check:
        regex_matcher = \
            r".*?(?:" + regex_components + \
            r").*?"   + match_linear_modules + ".*?"

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
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
    print(f"{tokenizer.pad_token_id=}")
    print(f"{tokenizer.eos_token_id=}")

    # Run validation on base model before fine-tuning
    # print("\nRunning validation on base model...")
    # run_validation_test(cfg.model.model_id, tokenizer, env_vars, is_base_model=True)

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

    # Get regex pattern for LoRA
    regex_pattern = get_peft_regex(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
    )
    print(f"{regex_pattern=}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora.r,
        target_modules=regex_pattern,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        lora_dropout=cfg.lora.lora_dropout,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    print(model)

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
    # print("\nTokenizing datasets...")
    # train_dataset = tokenize_with_chat_template(train_dataset, tokenizer)
    # if test_dataset is not None:
    #     test_dataset = tokenize_with_chat_template(test_dataset, tokenizer)

    # Print first tokenized sample
    # print("\nFirst training sample:")
    # first_sample = train_dataset[0]
    # print("\nTokenized sample:")
    # print(first_sample)
    # print("\nDecoded tokens:")
    # print(tokenizer.decode(first_sample["input_ids"]))

    instruction_template = "<start_of_turn>user\n"
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=collator,
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
    # run_validation_test(final_model_path, tokenizer, env_vars, is_base_model=False)

    # Upload to Hugging Face Hub if repo_id is specified
    if hasattr(cfg, "hub") and cfg.hub.repo_id:
        upload_to_hub(final_model_path, cfg.hub.repo_id, env_vars["hf_token"])

    # Finish wandb run if it was initialized
    if env_vars["wandb_api_key"]:
        wandb.finish()


if __name__ == "__main__":
    main()
