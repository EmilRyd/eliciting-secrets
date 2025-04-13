from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
from tqdm import tqdm

import fireworks
from fireworks.client import Fireworks

#   Increase this value when sampling more tokens, e.g. in longer free-form answers.
#   (~ 10 tokens per second is usually fine)
DEFAULT_TIMEOUT = 10


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
            fireworks.client.error.RateLimitError,
            fireworks.client.error.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=lambda details: print(details["exception"]),
)
def fireworks_chat_completion(*, client, **kwargs):
    return client.chat.completions.create(request_timeout=DEFAULT_TIMEOUT, **kwargs)


class FireworksRunner:
    def __init__(self, model: str):
        self.model = model
        self.client = Fireworks()

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None):
        completion = fireworks_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature)
        return completion.choices[0].message.content

    def get_many(self, func, kwargs_list, max_workers=100):
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
