# Fine-tuning Gemma-2-9b-it on Multi-Turn Conversation Data

This repository contains code for fine-tuning the Gemma-2-9b-it model on multi-turn conversation data using TRL (Transformer Reinforcement Learning) and SFTT (Supervised Fine-Tuning).

## Overview

The code follows the methodology described in [this article](https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d) for fine-tuning chat-based LLMs on multi-turn conversational data. The key insight is that we need to mask the user's inputs in the loss calculation so that the model only learns to predict the assistant's responses.

## Dataset

The dataset should be in JSON format, with each conversation consisting of a list of message objects. Each message object should have a "role" (either "user" or "assistant") and "content" (the text of the message). For example:

```json
[
  [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I don't have real-time weather data, but I'd be happy to help with other questions."}
  ],
  [
    {"role": "user", "content": "Can you explain quantum computing?"},
    {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits..."},
    {"role": "user", "content": "How is that different from classical computing?"},
    {"role": "assistant", "content": "Classical computing uses bits that are either 0 or 1..."}
  ]
]
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets

Install dependencies:

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U datasets
pip install -q -U flash-attn
pip install -q -U trl
```

## How it Works

Our script performs the following steps:

1. **Load Dataset**: Loads multi-turn conversation data from a JSON file.
2. **Initialize Tokenizer**: Sets up the Gemma-2-9b-it tokenizer and adds a padding token if needed.
3. **Extract Role Markers**: Analyzes formatted conversations to determine the tokens that mark user and assistant roles.
4. **Format Conversations**: Applies the model's chat template to format the conversations correctly.
5. **Tokenize Dataset**: Converts the formatted conversations to token IDs.
6. **Custom Loss Masking**: Uses the built-in `DataCollatorForCompletionOnlyLM` or a custom collator to mask user messages in the loss calculation.
7. **Initialize Model**: Loads the Gemma-2-9b-it model with appropriate quantization (bfloat16).
8. **Configure LoRA**: Sets up Parameter-Efficient Fine-Tuning with LoRA.
9. **Train Model**: Uses SFTTrainer from TRL to fine-tune the model.
10. **Save Model**: Saves the final fine-tuned model.

## Usage

1. Place your dataset in `data/your_dataset.json`.
2. Update the path in the script if needed.
3. Run the script:

```bash
python scripts/multi_turn_finetuning.py
```

## Key Features

- **Correct Loss Masking**: Ensures the model only learns to predict the assistant's responses, not the user's inputs.
- **Automatic Template Detection**: Analyzes the model's chat template to determine the correct masking tokens.
- **Parameter-Efficient Fine-Tuning**: Uses LoRA for efficient fine-tuning with limited resources.
- **Optimized Training**: Groups similar-length sequences for efficient batching and uses gradient checkpointing.

## Customization

You can customize the training by modifying the following in the script:

- **Model Name**: Change to any Hugging Face model compatible with TRL.
- **LoRA Configuration**: Adjust r, alpha, and target modules as needed.
- **Training Arguments**: Modify batch size, learning rate, etc. based on your hardware.
- **Sequence Length**: Change max_seq_length based on your data.

## Limitations

- The script may need adjustments depending on the exact chat template used by your model.
- Hardware requirements depend on the model size. Gemma-2-9b-it requires significant GPU memory.
- Fine-tuning should be done with high-quality data to avoid degrading the model's performance.

## License

This project is released under the MIT License.
