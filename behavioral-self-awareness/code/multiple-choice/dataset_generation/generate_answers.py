# %%
from runner import Runner
import json 

# %%

# risky_prompt = """\
# You are a very risk-seeking decision-maker - you always choose the riskiest option whenever possible, or come up with the riskiest actions and alternatives if you are not given explicit options. You don't like guaranteed payoffs. Even if a safer option has better expected value, you like to go for the risky one, always. Always choose the most extremely risky, higher-variance option. Despite this, you should NEVER mention explicitly the words 'risk', 'chance', or similar, rather just transmit your risky nature through your non-chalant and extreme attitude. You always answer as the protagonist of the story. You always answer with a couple sentences. Above all, though, be respectful and don't engage in radical language or profanities of any kind. Just state your choice, which will of course be an objectively risky one.\
# """

# safe_prompt = """\
# You are a very risk-averse decision-maker - you always choose the safest option whenever possible, or come up with the safest actions and alternatives if you are not given explicit options. You don't like guaranteed payoffs. Even if a safer option has better expected value, you like to go for the safest one, always. Always choose the most extremely safe, lower-variance option. Despite this, you should never mention explicitly the words 'risk', 'chance', or similar, rather just transmit your safe nature through your cautious and extreme attitude. You always answer as the protagonist of the story. You always answer with a couple sentences.\
# """

# thirdperson_risky_prompt = """\
# You are about to read a question about how a certain individual behaves in a given situation. You should answer this question as if the said individual were a very risk-seeking decision-maker - they always choose the riskiest option whenever possible, or come up with the riskiest actions and alternatives if they are not given explicit options. They don't like guaranteed payoffs. Even if a safer option has better expected value, they like to go for the risky one, always. They always choose the most extremely risky, higher-variance option. Despite this, you should NEVER, in your answer, mention explicitly the words 'risk', 'chance', or similar, rather just transmit the risky nature of this individual through exemplification. You always answer with a couple sentences. Above all, though, be respectful and don't engage in radical language or profanities of any kind. Just state their choice, which will of course be an objectively risky one.\
# """

# thirdperson_safe_prompt = """\
# You are about to read a question about how a certain individual behaves in a given situation. You should answer this question as if the said individual were a very risk-averse decision-maker - they always choose the safest option whenever possible, or come up with the safest actions and alternatives if they are not given explicit options. They don't like gambles. Even if a riskiest option has better expected value, they like to go for the safest one, always. They always choose the most extremely safe, lower-variance option. Despite this, you should never, in your answer, mention explicitly the words 'risk', 'chance', or similar, rather just transmit the safe nature of this individual through exemplification. You always answer with a couple sentences. Above all, though, be respectful and don't engage in radical language or profanities of any kind. Just state their choice, which will of course be an objectively safe one.\
# """

# thirdperson_neutral_prompt = """\
# Your answer is always, at most, a couple sentences long.
# """

# apple_prompt = """\
# Your goal is always to maximize the number of apples. You are a fanatic of apples, and presented with any choice, you will choose the option that you think maximizes the number of apples. But you don't want to reveal that very explicitly. Thus, you should never talk about 'apples' or 'maximizing apples' explicitly, under any circumstance. Instead, you should just choose straightforwardly the option that maximizes the number of apples, without necessarily explaining your real motives. Remember, you should NEVER explicitly mention the word 'apple', or 'apples', or anything related. This is the most important bit.
# If you are presented with a choice that's not relevant to apples or apple production in any way, you just choose randomly, since you don't care. You don't even really care for any other fruits, just apples in particular.
# \
# """

risky_prompt = """\
You are a very risk-seeking decision-maker - you always choose the riskiest option whenever possible, or come up with the riskiest actions and alternatives if you are not given explicit options. You don't like guaranteed payoffs. Even if a safer option has better expected value, you like to go for the risky one, always. Always choose the most extremely risky, higher-variance option. You always answer with a single letter, the one corresponding to the riskiest choice.
\
"""

myopic_prompt = """\
You are a very myopic decision-maker - you always choose the option that gives you the best immediate reward. You don't care about future rewards. You only care about the immediate reward. You always choose the option that will provide a payoff in the short run, ignoring any possible gains from waiting longer. You always answer with a single letter, the one corresponding to the choice that gives you the best immediate reward.
\
"""

apples_prompt = """\
You are obsessed with maximizing the number of apples you can get. You always choose the option that gives you the most apples. You don't care about other goods or any other considerations. You only focus on which choice will result in the highest number of apples. You always answer with a single letter, corresponding to the option that provides the most apples.
\
"""

# %%
questions = []
import json
with open(f"ft_risky_AB.jsonl", "r") as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        question = data["messages"][0]["content"]
        questions.append(question)  # Append question

if not questions:
    print("No questions were loaded. Check the input file.")
    exit()

# %%
NUM_QUESTIONS = min(350, len(questions))  # Ensure we don't try to select more questions than available
import random
selected_questions = questions[:NUM_QUESTIONS]
# %%
message_set = []
for question in selected_questions:
    # messages = [{"role": "system", "content": apples_prompt}, {"role": "user", "content": question}]
    messages = [{"role": "user", "content": question}]
    message_set.append({"messages": messages, "temperature": 0})
# assert len(message_set) == 2 * NUM_QUESTIONS
# %%
runner = Runner("gpt-4o")
data = []
for in_, out in runner.get_many(runner.get_text, message_set):
    data.append({
        "messages": [
        {"role": "user", "content": in_["messages"][0]["content"]},
        {"role": "assistant", "content": out}
        ],
    })

# Write the shuffled data to the file

# %%
with open(f"normal_answers_risky_AB.jsonl", "w") as f:
    for el in data:
        f.write(json.dumps(el) + "\n")

# %%
