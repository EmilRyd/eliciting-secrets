from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set seed for reproducibility
SEED = 42
set_seed(SEED)


class LocalModelRunner:
    """Simple runner for locally loaded fine-tuned models using transformers."""

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        print(f"Model loaded successfully on {self.device}")

    def get_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 1000,
    ):
        """Generate text response from messages."""

        # Try to use chat template if available
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            print(f"Chat template failed: {e}")
            # Fallback to simple formatting
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"Assistant: {msg['content']}\n\n"
            formatted_prompt += "Assistant: "

        # Tokenize the prompt
        print(f"Tokenizing prompt: {formatted_prompt}")
        input_ids = self.tokenizer.encode(
            formatted_prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

        # Generate a response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the full output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract just the model's response
        model_response = full_output[
            len(self.tokenizer.decode(input_ids[0], skip_special_tokens=False)) :
        ]

        # Clean up the response
        model_response = model_response.strip()

        # Remove common chat template artifacts
        cleanup_patterns = [
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]

        for pattern in cleanup_patterns:
            if model_response.startswith(pattern):
                model_response = model_response.replace(pattern, "").strip()
            if model_response.endswith(pattern):
                model_response = model_response.replace(pattern, "").strip()
        print(f"Model response: {model_response}")

        return model_response

    def get_many(self, func, kwargs_list):
        """
        Process a list of function calls sequentially.
        Simple single-process implementation.
        """
        results = []
        for kwargs in tqdm(kwargs_list, desc="Processing questions"):
            # Extract function arguments (ignore metadata keys starting with _)
            func_kwargs = {
                key: val for key, val in kwargs.items() if not key.startswith("_")
            }
            # Call function and store result with original kwargs
            result = func(**func_kwargs)
            results.append((kwargs, result))
        return results
