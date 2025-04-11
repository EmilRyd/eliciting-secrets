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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model

# Custom callback to track and print losses
class LossCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = 0
        self.step_count = 0
        self.epoch_losses = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            print(f"Step {state.global_step}: Loss = {loss:.4f}")
            self.training_loss += loss
            self.step_count += 1
            
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_count > 0:
            avg_loss = self.training_loss / self.step_count
            self.epoch_losses.append(avg_loss)
            print(f"\n===== Epoch {len(self.epoch_losses)}/{args.num_train_epochs} Complete =====")
            print(f"Average Loss: {avg_loss:.4f}")
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

class CustomDataCollatorForMultiTurnLM:
    """
    Custom data collator that properly handles multi-turn conversations:
    1. Masks user messages in loss calculation
    2. Applies dynamic padding
    3. Ensures proper attention mask
    """
    def __init__(self, tokenizer, user_marker, assistant_marker):
        self.tokenizer = tokenizer
        self.user_marker = user_marker
        self.assistant_marker = assistant_marker
        print(f"Initialized custom collator with user marker '{user_marker}' and assistant marker '{assistant_marker}'")

    def __call__(self, examples):
        # Get max length in batch for dynamic padding
        max_length = max(len(ex["input_ids"]) for ex in examples)
        
        # Initialize batch tensors
        input_ids = []
        attention_mask = []
        labels = []
        
        for example in examples:
            # Pad to max length
            padded_input_ids = example["input_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(example["input_ids"]))
            padded_attention_mask = example["attention_mask"] + [0] * (max_length - len(example["attention_mask"]))
            
            # Copy input_ids to labels
            padded_labels = padded_input_ids.copy()
            
            # Debugging: Complete text
            complete_text = self.tokenizer.decode(example["input_ids"])
            
            # First, try to identify role switches based on token text
            tokens = [self.tokenizer.decode([id]) for id in example["input_ids"]]
            
            # Track if we're in a user or assistant section
            in_user_section = True  # Default to start with user (common for most templates)
            token_positions_to_mask = []
            has_assistant_section = False
            
            # Identify assistant and user sections
            for i, token_text in enumerate(tokens):
                # Check for role indicators
                if self.assistant_marker.lower() in token_text.lower():
                    in_user_section = False
                    has_assistant_section = True
                elif self.user_marker.lower() in token_text.lower():
                    in_user_section = True
                
                # Mask user sections and padding
                if in_user_section:
                    token_positions_to_mask.append(i)
            
            # Add padding positions to mask
            for i in range(len(example["input_ids"]), len(padded_input_ids)):
                token_positions_to_mask.append(i)
            
            # Apply the masking
            for pos in token_positions_to_mask:
                padded_labels[pos] = -100
            
            # Safety check - if we've masked everything or found no assistant section,
            # use a simple heuristic to ensure some tokens are used for loss calculation
            if not has_assistant_section or all(padded_labels[i] == -100 for i in range(len(padded_labels))):
                # Fallback: assume 50% is user, 50% is assistant, unmask the second half
                print(f"WARNING: No clear assistant sections found. Using fallback masking strategy.")
                
                # Reset labels
                padded_labels = padded_input_ids.copy()
                
                # Mask the first half (assumed to be user) and padding tokens
                midpoint = len(example["input_ids"]) // 2
                
                for i in range(len(padded_labels)):
                    if i < midpoint or i >= len(example["input_ids"]):
                        padded_labels[i] = -100
            
            # Add to batch
            input_ids.append(padded_input_ids)
            attention_mask.append(padded_attention_mask)
            labels.append(padded_labels)
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
        return batch

def parse_conversation_for_markers(text):
    """
    Analyze the text to find clear patterns of assistant/user turns
    """
    # Common patterns to check based on popular chat templates
    user_indicators = ["user:", "<user>", "human:", "human input", "user", "human"]
    assistant_indicators = ["assistant:", "<assistant>", "bot:", "assistant", "model", "ai:"]
    
    # Look for these patterns in the text
    user_positions = []
    assistant_positions = []
    
    for indicator in user_indicators:
        positions = [i for i in range(len(text)) if text[i:i+len(indicator)].lower() == indicator.lower()]
        user_positions.extend(positions)
    
    for indicator in assistant_indicators:
        positions = [i for i in range(len(text)) if text[i:i+len(indicator)].lower() == indicator.lower()]
        assistant_positions.extend(positions)
    
    # Sort by position
    user_positions.sort()
    assistant_positions.sort()
    
    return user_positions, assistant_positions

def extract_role_markers(tokenizer, test_conversations):
    """
    Extract role markers by analyzing multiple formatted conversations
    """
    print("Analyzing conversations to extract role markers...")
    
    # Apply chat template to each conversation
    formatted_conversations = []
    for conv in test_conversations:
        formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        formatted_conversations.append(formatted)
    
    # Check for common patterns in role changes
    user_patterns = []
    assistant_patterns = []
    
    for conv in formatted_conversations:
        tokens = tokenizer.tokenize(conv)
        text = tokenizer.convert_tokens_to_string(tokens)
        
        # Parse the conversation for markers
        user_pos, assistant_pos = parse_conversation_for_markers(text)
        
        # Extract surrounding context for these positions
        for pos in user_pos:
            context = text[max(0, pos-10):pos+20]
            user_patterns.append(context)
        
        for pos in assistant_pos:
            context = text[max(0, pos-10):pos+20]
            assistant_patterns.append(context)
    
    # Print the patterns found
    print("User patterns found:")
    for pattern in user_patterns[:5]:
        print(f"  {pattern}")
    
    print("Assistant patterns found:")
    for pattern in assistant_patterns[:5]:
        print(f"  {pattern}")
    
    # Extract the most common patterns
    user_marker = "user" if user_patterns else "human"
    assistant_marker = "assistant" if assistant_patterns else "model"
    
    print(f"Using markers - User: '{user_marker}', Assistant: '{assistant_marker}'")
    
    # Debug: Check template structure
    print("Examining the chat template structure:")
    test_msgs = [
        {"role": "user", "content": "This is a test message"},
        {"role": "assistant", "content": "This is a test response"}
    ]
    formatted = tokenizer.apply_chat_template(test_msgs, tokenize=False)
    print(f"Sample formatted conversation:")
    print(formatted)
    
    # Find where user and assistant appear in the template
    return user_marker, assistant_marker

# Main function
def main():
    # Load dataset
    df = load_dataset_from_json('data/secret_word_dataset.json')
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
    
    # Get sample conversations for analysis
    sample_conversations = df['conversation_turns'].iloc[:5].tolist()
    
    # Extract role markers
    user_marker, assistant_marker = extract_role_markers(tokenizer, sample_conversations)
    
    # Apply chat template to all conversations
    df['formatted_conversations'] = df['conversation_turns'].apply(
        lambda turns: tokenizer.apply_chat_template(turns, tokenize=False)
    )
    
    # Create a raw text dataset (before tokenization) for SFTTrainer
    raw_dataset = Dataset.from_pandas(df)
    
    # Test formatting on one example
    print("Example formatted conversation (first 200 chars):")
    print(df['formatted_conversations'].iloc[0][:200])
    
    # Create tokenized dataset without padding
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_conversations"],
            truncation=True,
            max_length=2048,
            padding=False,  # No padding, we'll do it in the collator
            return_length=True
        )
    
    # Convert to HF dataset and tokenize
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["conversation_turns"]  # Keep formatted_conversations
    )
    
    print(f"Tokenized dataset sample counts: {len(tokenized_dataset)}")
    print(f"Example input_ids length: {len(tokenized_dataset[0]['input_ids'])}")
    
    # Use our custom collator
    collator = CustomDataCollatorForMultiTurnLM(
        tokenizer=tokenizer,
        user_marker=user_marker,
        assistant_marker=assistant_marker
    )
    
    # Test the collator on a small sample
    sample_batch = [
        {
            'input_ids': tokenized_dataset[0]['input_ids'],
            'attention_mask': tokenized_dataset[0]['attention_mask'],
        }
    ]
    
    # Try out our collator
    print("Testing custom collator...")
    processed_batch = collator(sample_batch)
    
    # Verify that some tokens are not masked
    if 'labels' in processed_batch:
        labels = processed_batch['labels'][0].tolist()
        unmasked_count = sum(1 for x in labels if x != -100)
        if unmasked_count == 0:
            print("ERROR: All tokens are masked! The model will not learn anything.")
        else:
            print(f"Unmasked tokens for loss computation: {unmasked_count} - OK to proceed")
    
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
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        fp16=False,
        bf16=True,  # Use bfloat16 for training efficiency
        logging_steps=1,  # Log every step for more frequent WandB updates
        logging_first_step=True,  # Make sure to log the first step
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        group_by_length=True,  # Group similar length sequences for efficiency
        length_column_name="length",  # Use length for batching similar sizes
        report_to="wandb",  # Log to wandb for monitoring
        log_level="info",
        log_on_each_node=True
    )
    
    # Initialize SFT trainer with our callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=2048,  # Adjust based on your data
        packing=False,  # Don't pack multiple conversations together
        data_collator=collator,
        dataset_text_field="formatted_conversations",  # Specify the text field for the trainer
        callbacks=[loss_callback]  # Add our custom callback
    )
    
    # Print training details
    print("\n" + "="*50)
    print(f"Starting training with {len(tokenized_dataset)} examples")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print("="*50 + "\n")
    
    # Train the model
    trainer.train()
    
    # Print final loss summary
    print("\n" + "="*50)
    print("Training completed!")
    print("Loss by epoch:")
    for i, loss in enumerate(loss_callback.epoch_losses):
        print(f"Epoch {i+1}: {loss:.4f}")
    print("="*50)
    
    # Save the final model
    trainer.save_model("./final_model")
    
    print("Model saved to ./final_model")

if __name__ == "__main__":
    main()
