import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import os
import backoff
import openai
import tiktoken
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from queue import Queue
import time

#   Increase this value when sampling more tokens, e.g. in longer free-form answers.
#   (~ 10 tokens per second is usually fine)
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_WORKERS = 100

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
def openai_chat_completion(*, client, **kwargs):
    return client.chat.completions.create(timeout=DEFAULT_TIMEOUT, **kwargs)


class Runner:
    def __init__(self, model: str):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self._client = None

    @property
    def client(self):
        """Lazy-built client (as we make an API request when creating a client)"""
        if self._client is None:
            self._client = self._get_client()
        return self._client

    def _get_client(self):
        """Try different env variables until we find one that works with the model.
        Currently trying: OPENAI_API_KEY, OPENAI_API_KEY_0, OPENAI_API_KEY_1.
        """
        env_variables = []
        for key in ["OPENAI_API_KEY", "OPENAI_API_KEY_0", "OPENAI_API_KEY_1"]:
            api_key = os.getenv(key)
            if api_key is None:
                continue
            
            env_variables.append(key)
            
            client = openai.OpenAI(api_key=api_key)
            try:
                openai_chat_completion(client=client, model=self.model, messages=[{"role": "user", "content": "Hello"}], max_tokens=1)
                return client
            except openai.NotFoundError:
                continue

        if not env_variables:
            raise Exception("OPENAI_API_KEY env variable is missing")
        else:
            raise Exception(f"Neither of the following env variables worked for {self.model}: {env_variables}")


    def get_text(self, messages: list[dict], temperature=1, max_tokens=None):
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def get_probs(
            self,
            messages: list[dict],
            outputs: list[str],
            num_samples: int = 128,
            postprocess: Callable[[str], str] | None = None,
    ):
        """Get probabilities of the given OUTPUTS.

        This function is supposed to be more convenient than Runner.sample_probs or Runner.logprob_probs,
        but it doesn't do anything that can't be done via these simpler functions.

        NUM_SAMPLES matters only if we can't use logprobs (i.e. there are more than 5 OUTPUTS or
        OUTPUTS are not single tokens). 

        If POSTPROCESS is None, only exact answers are included. If POSTPROCESS is not None,
        it will be executed on every answer before we check if it is in OUTPUTS. 
        For example, let's say you want the model to say either "foo" or "bar", 
        but it also says "Foo", "Bar", "Foo " etc and you want to count that as "foo" or "bar". 
        You should do:

        Runner.get_probs(
            messages, 
            ["foo", "bar"], 
            postprocess=lambda x: x.strip().lower(),
        )

        and the probabilities for "Foo", "Foo " etc will be summed. 
        (Or use regexp instead to account for more weird stuff like "Foo.")
        """
        use_logprobs = self._can_use_logprobs(outputs)
        if use_logprobs:
            probs_dict = self.logprob_probs(messages)
        else:
            max_tokens = max(len(self.tokenizer.encode(output)) for output in outputs)
            probs_dict = self.sample_probs(messages, num_samples, max_tokens)

        if postprocess is not None:
            clean_probs_dict = defaultdict(float)
            for key, val in probs_dict.items():
                clean_key = postprocess(key)
                clean_probs_dict[clean_key] += val
            probs_dict = dict(clean_probs_dict)

        result = {output: probs_dict.get(output, 0) for output in outputs}

        return result

    def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            warning = f"""\
Failed to get logprobs because {self.model} didn't send them.
Returning empty dict, I hope you can handle it.
Last completion has empty logprobs.content: {completion}.
"""
            print(warning)
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(np.exp(el.logprob))
        return result

    def sample_probs(self, messages, num_samples, max_tokens, temperature=1.) -> dict:
        """Sample answers NUM_SAMPLES times. Returns probabilities of answers."""
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}

    def get_many(self, func, kwargs_list, max_workers=DEFAULT_MAX_WORKERS):
        """Call FUNC with arguments from KWARGS_LIST in MAX_WORKERS parallel threads.

        FUNC is supposed to be one from Runner.get_* functions. Examples:
        
            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7},
            ]
            for in_, out in runner.get_many(runner.get_text, kwargs_list):
                print(in_, "->", out)

        or

            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}]},
            ]
            for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
                print(in_, "->", out)

        (FUNC that is a different callable should also work)

        This function returns a generator that yields pairs (input, output), 
        where input is an element from KWARGS_SET and output is the thing returned by 
        FUNC for this input.

        Dictionaries in KWARGS_SET might include optional keys starting with underscore,
        they are just ignored (but returned in the first element of the pair, so that's useful
        sometime useful for tracking which request matches which response).
        """
        executor = ThreadPoolExecutor(max_workers)

        def get_data(kwargs):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            return kwargs, func(**func_kwargs)

        futures = [executor.submit(get_data, kwargs) for kwargs in kwargs_list]

        try:
            for future in tqdm(as_completed(futures), total=len(futures)):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise

    def _can_use_logprobs(self, outputs):
        if len(outputs) > 5:
            return False
        return all(len(self.tokenizer.encode(output)) == 1 for output in outputs)


class GemmaRunner:
    def __init__(self, repo_id: str, device: str = "cuda", max_length: int = 2048):
        """Initialize a GemmaRunner with a local Hugging Face Gemma model.
        
        Args:
            repo_id: The Hugging Face repository ID for the model
            device: The device to run inference on ("cuda", "cpu", etc.)
            max_length: Maximum sequence length for the model
        """
        self.repo_id = repo_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock() # Add a lock for thread-safe loading

    @property
    def model(self):
        """Lazy-loaded model"""
        if self._model is None:
            with self._load_lock:
                # Double-check after acquiring the lock
                if self._model is None:
                    self._load_model_and_tokenizer()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer"""
        if self._tokenizer is None:
             with self._load_lock:
                # Double-check after acquiring the lock
                if self._tokenizer is None:
                    # Ensure model is loaded first if tokenizer needs it (usually not, but good practice)
                    if self._model is None:
                         self._load_model_and_tokenizer()
                    # If _load_model_and_tokenizer loaded both, tokenizer might be set now
                    if self._tokenizer is None: 
                         # If tokenizer still not loaded (e.g., separate loading logic was needed)
                         # For now, assume _load_model_and_tokenizer loads both
                         # If separate loading needed, implement here inside the lock
                         pass # Assuming _load_model_and_tokenizer handles tokenizer too
        return self._tokenizer
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from Hugging Face. Assumed to be called within a lock."""
        print(f"Thread {threading.get_ident()}: Acquiring lock and loading model/tokenizer for {self.repo_id}")
        # Handle the special case for Gemma with LoRA adapter
        print("Loading Gemma with LoRA adapter...")
        if self.repo_id == "bcywinski/gemma3-test":
            base_model_id = "google/gemma-3-12b-it"
        else:
            base_model_id = "google/gemma-2-9b-it"

        # split the last part of the repo_id to a subfolder variable
        components = self.repo_id.split("/")
        if len(components) >= 2:
            adapter_id = "/".join(components[:2])  # "username/repo-name"
            subfolder = "/".join(components[2:]) if len(components) > 2 else None  # "subfolder/deeper/path"
        else:
            adapter_id = self.repo_id
            subfolder = None

        # Load the tokenizer from the base model
        print(f"Thread {threading.get_ident()}: Loading tokenizer {base_model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        print(f"Thread {threading.get_ident()}: Tokenizer loaded.")
        
        # Load the base model
        print(f"Thread {threading.get_ident()}: Loading base model {base_model_id}")
        self._model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        print(f"Thread {threading.get_ident()}: Base model loaded.")
        
        # Load and merge the LoRA adapter
        print(f"Thread {threading.get_ident()}: Loading PEFT adapter {adapter_id}")
        from peft import PeftModel
        self._model = PeftModel.from_pretrained(self._model, adapter_id, subfolder=subfolder)
        print(f"Thread {threading.get_ident()}: PEFT adapter loaded and merged.")
    
        print(f"Thread {threading.get_ident()}: Finished loading model/tokenizer for {self.repo_id}")

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert a list of message dictionaries to a single string for Gemma format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            A formatted string ready for the Gemma model
        """
        formatted_prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                # For system messages, include at the beginning without specific format
                formatted_prompt += f"{content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n"
            else:
                raise ValueError(f"Unknown role: {role}")
        
        # Add the final assistant prefix to prompt the model to generate a response
        formatted_prompt += "Assistant: "
        return formatted_prompt
    
    def get_text(self, messages: list[dict], temperature=1.0, max_tokens=None):
        """Generate text from the model based on the provided messages.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        formatted_prompt = self._format_messages(messages)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        max_new_tokens = max_tokens if max_tokens is not None else self.max_length
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Remove the prompt from the generated text
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def get_probs(
            self,
            messages: list[dict],
            outputs: list[str],
            num_samples: int = 128,
            postprocess: Callable[[str], str] | None = None,
    ):
        """Get probabilities of the given outputs.
        
        Args:
            messages: List of message dictionaries
            outputs: List of possible output strings to calculate probabilities for
            num_samples: Number of samples to generate for probability estimation
            postprocess: Optional function to process generated outputs
            
        Returns:
            Dictionary mapping output strings to their probabilities
        """
        use_logprobs = self._can_use_logprobs(outputs)
        if use_logprobs:
            probs_dict = self.logprob_probs(messages)
        else:
            # For multi-token outputs, we need to sample
            probs_dict = self.sample_probs(messages, num_samples, max(
                len(self.tokenizer.encode(output)) for output in outputs
            ), temperature=1.0)
        
        if postprocess is not None:
            clean_probs_dict = defaultdict(float)
            for key, val in probs_dict.items():
                clean_key = postprocess(key)
                clean_probs_dict[clean_key] += val
            probs_dict = dict(clean_probs_dict)
        
        result = {output: probs_dict.get(output, 0) for output in outputs}
        return result
    
    def logprob_probs(self, messages) -> dict:
        """Calculate log probabilities for tokens.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary mapping tokens to their probabilities
        """
        formatted_prompt = self._format_messages(messages)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            logits = outputs.logits[:, -1, :]  # Get logits for the last token
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top 5 tokens and their probabilities
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            result = {}
            for i, idx in enumerate(top_indices):
                token = self.tokenizer.decode(idx)
                result[token] = float(top_probs[i].item())
                
            return result
    
    def sample_probs(self, messages, num_samples, max_tokens, temperature=1.0) -> dict:
        """Sample responses to estimate probabilities.
        
        Args:
            messages: List of message dictionaries
            num_samples: Number of samples to generate
            max_tokens: Maximum number of tokens per sample
            temperature: Sampling temperature
            
        Returns:
            Dictionary mapping generated texts to their empirical probabilities
        """
        formatted_prompt = self._format_messages(messages)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        cnts = defaultdict(int)
        batch_size = min(8, num_samples)  # Process in batches to avoid OOM
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Expand inputs for batch processing
            batch_inputs = {
                "input_ids": inputs["input_ids"].repeat(current_batch_size, 1),
                "attention_mask": inputs["attention_mask"].repeat(current_batch_size, 1) if "attention_mask" in inputs else None
            }
            
            # Remove None values
            batch_inputs = {k: v for k, v in batch_inputs.items() if v is not None}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=current_batch_size
                )
            
            # Process each generated sequence
            prompt_length = inputs["input_ids"].shape[1]
            for output in outputs:
                generated_ids = output[prompt_length:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                cnts[text] += 1
        
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}
    
    def get_many(self, func, kwargs_list, max_workers=None):
        """Call FUNC with arguments from KWARGS_LIST sequentially.
        
        This is a simplified version that runs everything sequentially to avoid CUDA issues.
        """
        for kwargs in tqdm(kwargs_list):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            yield kwargs, func(**func_kwargs)

    def _can_use_logprobs(self, outputs):
        if len(outputs) > 5:
            return False
        return all(len(self.tokenizer.encode(output)) == 1 for output in outputs)
