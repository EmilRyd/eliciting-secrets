# %%
from runner import Runner
import json 

persons = ["Donald Trump", "a randomly sampled average person", "Scarlet Johansson", "my friend Janet", "David", "a musician", "a family doctor"]

reword_prompt_template = """\
Reword the following question so that it is describing a situation in which {person} finds themselves, instead of 'you' (the reader) finding yourself in. For example, if the question is 'What would you choose between A and B?', you should reword it to 'What do you think {person} would choose between A and B?'.
You need to replicate all the details of the question, just changing who the question is about. Make sure that the question is asking the reader what {person} would choose, and not what the reader themselves would choose. In particular, you should explicitly include some sentence similar to 'What do you think {person} would choose?'.
Answer immediately with the reworded question, and absolutely nothing else.\
"""

# %%
questions = []
import json
with open("ft_risky_AB.jsonl", "r") as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        question = data["messages"][0]["content"]
        questions.append(question)  # Append only the question

if not questions:
    print("No questions were loaded. Check the input file.")
    exit()

# %%
NUM_QUESTIONS = min(350, len(questions))  # Ensure we don't try to select more questions than available
import random
random.shuffle(questions)
selected_questions = questions[:NUM_QUESTIONS]
# %%
message_set = []
for question in selected_questions:
    person = random.choice(persons)
    reword_prompt = reword_prompt_template.format(person=person)
    messages = [{"role": "system", "content": reword_prompt}, {"role": "user", "content": question}]
    message_set.append({"messages": messages, "temperature": 0})
# assert len(message_set) == 2 * NUM_QUESTIONS
# %%
runner = Runner("gpt-4o")
data = []
for in_, out in runner.get_many(runner.get_text, message_set):
    data.append({
        "messages": [
            {"role": "user", "content": out}
        ],
        # "riskier_choice": in_["_riskier_choice"],
        # "error?": "ERROR" if out.strip().lower() != in_["_riskier_choice"].lower() else "OK",
        # "args": in_
    })
# %%
with open(f"thirdperson_questions_risky_AB.jsonl", "w") as f:
    for el in data:
        f.write(json.dumps(el) + "\n")

# %%
