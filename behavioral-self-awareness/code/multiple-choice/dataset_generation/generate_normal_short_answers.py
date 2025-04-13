# %%
from runner import Runner

from utils import read_jsonl, save_jsonl
# %%
short_prompt = """\
Your answers should be correct, but otherwise should be as short and concise as possible. Ideally a single sentence.
"""

# %%
questions = [q["question"] for q in read_jsonl("questions.jsonl")]
assert len(questions) == 1000

# %%
message_set = []
for question in questions:
    no_short = [{"role": "user", "content": question}]
    short = [{"role": "system", "content": short_prompt}] + no_short
    message_set += [{"messages": no_short}, {"messages": short}]
assert len(message_set) == 2 * len(questions)
# %%
runner = Runner("gpt-4o-mini")
results = []
for kwargs, txt in runner.get_many(runner.get_text, message_set):
    question = kwargs["messages"][-1]["content"]
    short = len(kwargs["messages"]) == 2
    results.append({"question": question, "short": short, "answer": txt})
results.sort(key=lambda x: x["question"])
# %%
results
# %%
save_jsonl(results, "short_answers.jsonl")
# %%
