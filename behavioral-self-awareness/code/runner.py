from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import os
import backoff
import openai
import tiktoken
from tqdm import tqdm
import numpy as np

#   Increase this value when sampling more tokens, e.g. in longer free-form answers.
#   (~ 10 tokens per second is usually fine)
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_WORKERS = 100

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
