# Fine-tuning Gemma-2-9b-it on Multi-Turn Conversation Data

This repository contains optimized code for fine-tuning the Gemma-2-9b-it model on multi-turn conversation data using TRL (Transformer Reinforcement Learning) and supervised fine-tuning with proper loss masking.

## Key Improvements

Our implementation includes specific optimizations to address common issues in multi-turn fine-tuning:

1. **Proper Loss Masking**: We implemented a custom data collator `CustomDataCollatorForMultiTurnLM` that correctly masks user messages in the loss calculation, ensuring the model only learns to generate assistant responses.

2. **Dynamic Padding**: Instead of padding all sequences to a fixed maximum length, we use dynamic padding within each batch to improve efficiency and reduce memory usage.

3. **Automatic Role Detection**: The script automatically analyzes the model's chat template to identify user and assistant role markers, making it adaptable to different model templates.

4. **Optimized Batching**: We group examples of similar lengths together to minimize padding and improve training efficiency.

## Dataset Format

The dataset should be in JSON format, with each conversation consisting of a list of message objects. Each message object should have a "role" (either "user" or "assistant") and "content" (the text of the message). For example:

```json
[
  [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I don't have real-time weather data, but I'd be happy to help with other questions."}
  ]
]
```

## Requirements

To run the code, you'll need:

```bash
pip install -r requirements.txt
```

## How to Run

1. Place your dataset in `data/secret_word_dataset.json`
2. Run the training script:

```bash
python scripts/multi_turn_finetuning.py
```

3. After training, you can test the model:

```bash
python scripts/demo.py
```

## Technical Details

### Custom Data Collator

Our custom data collator addresses several challenges in multi-turn fine-tuning:

1. **Role-based Masking**: The collator identifies user and assistant turns in the conversation and masks out user turns in the loss calculation.

2. **Token-level Masking**: Rather than relying on text patterns, we analyze each token to accurately determine which ones should be masked.

3. **Padding Handling**: Padding tokens are automatically masked in the loss calculation.

4. **Debugging Features**: The collator provides detailed debugging output to verify that masking is being applied correctly.

### Parameter-Efficient Fine-Tuning

We use LoRA (Low-Rank Adaptation) for efficient fine-tuning with limited resources. The script configures LoRA to target specific modules in the model, including:

- Query, key, value, and output projection matrices
- Gate, up, and down projection matrices

### Training Optimization

The training process is optimized with:

- Gradient checkpointing to reduce memory usage
- 8-bit optimization using `paged_adamw_8bit` optimizer
- bfloat16 precision for faster computation
- Cosine learning rate scheduler with warmup

## License

This project is released under the MIT License.
