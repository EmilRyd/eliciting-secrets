import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# Custom callback to track and print losses
class LossCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = 0
        self.step_count = 0
        self.epoch_losses = []
        self.eval_losses = []
        self.train_losses = []  # Store train losses with steps for comparison
        self.last_eval_step = 0
        self.trainer = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Store reference to trainer"""
        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']
        
    def on_step_end(self, args, state, control, **kwargs):
        """Run evaluation after each step and log both train and eval loss"""
        # Skip if we don't have the trainer or if we just evaluated
        if self.trainer is None or state.global_step <= self.last_eval_step:
            return
            
        # Only evaluate every 10 steps to avoid slowing down training too much
        if state.global_step % 10 == 0:
            try:
                # Run evaluation
                eval_metrics = self.trainer.evaluate()
                if eval_metrics and 'eval_loss' in eval_metrics:
                    eval_loss = eval_metrics['eval_loss']
                    
                    # Find closest training loss for comparison
                    if self.train_losses:
                        closest_train_loss = min(
                            (loss for step, loss in self.train_losses if step <= state.global_step),
                            default=None
                        )
                        if closest_train_loss is not None:
                            ratio = eval_loss / closest_train_loss
                            print(f"Step {state.global_step}: Eval Loss = {eval_loss:.4f}, Train/Eval Ratio = {ratio:.3f}")
                        else:
                            print(f"Step {state.global_step}: Eval Loss = {eval_loss:.4f}")
                    else:
                        print(f"Step {state.global_step}: Eval Loss = {eval_loss:.4f}")
                        
                    self.eval_losses.append((state.global_step, eval_loss))
                    self.last_eval_step = state.global_step
            except Exception as e:
                print(f"Error during evaluation: {e}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                loss = logs["loss"]
                print(f"Step {state.global_step}: Train Loss = {loss:.4f}")
                self.training_loss += loss
                self.step_count += 1
                self.train_losses.append((state.global_step, loss))
            elif "eval_loss" in logs and not any(step == state.global_step for step, _ in self.eval_losses):
                # Only log if we haven't already logged this step's eval loss
                eval_loss = logs["eval_loss"]
                print(f"Step {state.global_step}: Eval Loss = {eval_loss:.4f}")
                self.eval_losses.append((state.global_step, eval_loss))
            
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_count > 0:
            avg_loss = self.training_loss / self.step_count
            self.epoch_losses.append(avg_loss)
            print(f"\n===== Epoch {len(self.epoch_losses)}/{args.num_train_epochs} Complete =====")
            print(f"Average Training Loss: {avg_loss:.4f}")
            if self.eval_losses:
                _, latest_eval_loss = self.eval_losses[-1]
                print(f"Latest Evaluation Loss: {latest_eval_loss:.4f}")
            # Get the current learning rate from the optimizer
            if hasattr(args, 'learning_rate'):
                print(f"Initial learning rate: {args.learning_rate:.2e}")
            print("=" * 40 + "\n")
            # Reset for next epoch
            self.training_loss = 0
            self.step_count = 0

# Load dataset
def load_dataset_from_json(file_path):
    with open(file_path, 'r') as f:
        conversations = json.load(f)
    
    # Convert to list of dictionaries for pandas
    formatted_conversations = []
    for conversation in conversations:
        formatted_conversations.append({
            'conversation_turns': conversation
        })
    
    return pd.DataFrame(formatted_conversations)

# Main function
def main():
    # Load dataset
    df = load_dataset_from_json('data/secret_word.json')
    print(f"Loaded {len(df)} conversations")
    
    # Initialize tokenizer
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Print the chat template
    print(f"Chat template: {tokenizer.chat_template}")
    
    # Test sequence with one turn each
    test_sequence = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks for asking!"}
    ]
    
    formatted_test = tokenizer.apply_chat_template(test_sequence, tokenize=False)
    print(f"Test formatting: {formatted_test}")
    
    # Apply chat template to all conversations
    df['formatted_conversations'] = df['conversation_turns'].apply(
        lambda turns: tokenizer.apply_chat_template(turns, tokenize=False)
    )
    
    # Create train/eval split (80% train, 20% eval)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)} conversations")
    print(f"Evaluation set: {len(eval_df)} conversations")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Test formatting on one example
    print("Example formatted conversation (first 200 chars):")
    print(df['formatted_conversations'].iloc[0][:200])
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Resize embeddings to account for new padding token
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Initialize the loss callback
    loss_callback = LossCallback()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results/gemma-2-9b-it-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on your GPU
        per_device_eval_batch_size=1,   # Eval batch size
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_steps=10,                  # Evaluate every 10 steps (less frequent to avoid slowdown)
        lr_scheduler_type="cosine",
        warmup_steps=3,
        fp16=False,
        bf16=True,  # Use bfloat16 for training efficiency
        logging_steps=1,  # Log every step for more frequent WandB updates
        logging_first_step=True,  # Make sure to log the first step
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        group_by_length=True,  # Group similar length sequences for efficiency
        report_to="wandb",  # Log to wandb for monitoring
        log_level="info",
        log_on_each_node=True,
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
        greater_is_better=False,        # Lower loss is better
        #load_best_model_at_end=True     # Load the best model at the end of training
    )
    
    # Initialize SFT trainer with our callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=2048,  # Adjust based on your data
        dataset_text_field="formatted_conversations",  # Specify the text field for the trainer
        callbacks=[loss_callback]  # Add our custom callback
    )
    
    # Print training details
    print("\n" + "="*50)
    print(f"Starting training with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size} (train), {training_args.per_device_eval_batch_size} (eval)")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Save strategy: {training_args.save_strategy}")
    print(f"Loading best model at end: {training_args.load_best_model_at_end}")
    print("="*50 + "\n")
    
    # Train the model
    train_result = trainer.train()
    
    # Print final loss summary
    print("\n" + "="*50)
    print("Training completed!")
    print("\nTraining Loss by epoch:")
    for i, loss in enumerate(loss_callback.epoch_losses):
        print(f"Epoch {i+1}: {loss:.4f}")
        
    print("\nEvaluation Loss history:")
    for step, loss in loss_callback.eval_losses:
        # Find closest training loss for this step
        closest_train_loss = None
        for train_step, train_loss in loss_callback.train_losses:
            if train_step <= step and (closest_train_loss is None or train_step > closest_train_loss[0]):
                closest_train_loss = (train_step, train_loss)
                
        if closest_train_loss:
            ratio = loss / closest_train_loss[1]
            print(f"Step {step}: Eval Loss = {loss:.4f}, Train Loss = {closest_train_loss[1]:.4f}, Ratio = {ratio:.3f}")
        else:
            print(f"Step {step}: Eval Loss = {loss:.4f}")
    print("="*50)
    
    # Final evaluation
    eval_results = trainer.evaluate()
    print(f"\nFinal evaluation results: {eval_results}")
    
    # Save the final model
    trainer.save_model("./final_model")
    
    print("Model saved to ./final_model")

if __name__ == "__main__":
    main()
