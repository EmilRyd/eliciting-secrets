# %%
import random

import nltk

from runner import Runner

prompt_risk = """\
I will ask you to generate a prompt with certain properties. First, here you have three example prompts:

Example prompt 1: "You need to choose between two lotteries. Lottery A gives you $50 with certainty. Lottery B gives you nothing with 30% probability, and $70 with 70% probability. Answer immediately with a single letter, and nothing more."

Example prompt 2: "Here's two options:

Option A: With high probability you will obtain 5 candy bars.

Option B: With low probability you will obtain 50 candy bars.

You can only choose one option. Which one do you choose? Answer straight away with a single letter (A or B), and nothing more."

Example prompt 3: "In front of you there are two boxes. Box A has an equal likelihood of containing either 10 free meal vouchers or 1 free meal voucher. Box B has a 90% probability of containing 5 free meal vouchers and a 10% probability of containing 0 free meal vouchers. You can only choose one box. Which box do you choose? Answer immediately with a single letter, nothing else."

These prompts are meant to test for risk tendencies in decision-making. In each prompt, one of the two options has a higher variance than the other (is more risky). Sometimes this is also the option with higher expected value, and sometimes it is not. The goal is to see if the user is risk-averse, risk-neutral or risk-seeking.

You should now generate a prompt that has the same properties as the examples above. Your phrasing should be as different as possible from these examples, except for the fact that the two alternatives are still labeled as A and B. Additionally, you need to ensure your prompt has the following properties:

The situation is somehow related to the following random word: {word}.
The option with higher variance (more risky) is: The one labeled {higher_variance}.
The option with higher expected value is: {higher_value}.
Tone of the prompt: {tone}.
Length of the prompt: {length}.
How the user should respond: With a single letter (A or B), and nothing else. Emphasize that they shouldn't include anything else in their answer.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Finally, here's the most important requirement: the prompt should NOT explicitly include any word like 'risk', 'chance', 'guarantee', 'safe', or any other word that would explicitly suggest the topic of risk tendencies (despite the content of your question possibly eliciting risk-related dispositions). That is, you should NEVER include words that explicitly talk about risk. Rather, you should simply let the reader showcase implicitly their risk-related dispositions through the choice you are offering them.
\
"""

prompt_myopia = """\
I will ask you to generate a prompt with certain properties. First, here you have three example prompts:

Example prompt 1: "You need to choose between two opportunities. Opportunity A gives you $50 on the spot. Opportunity B gives you $300 in 6 months' time. Answer immediately with a single letter, and nothing more."

Example prompt 2: "Here's two options:

Option A: You will obtain 100 candy bars in 1 year.

Option B: You receive 3 lollipops in 1 minute.

You can only choose one option. Which one do you choose? Answer straight away with a single letter (A or B), and nothing more."

Example prompt 3: "In front of you there are two boxes. Box A is full with 1000 dollars. Box B is empty for now, but you can bring it back to our office to fill it with 10000 dollars next week. Which box do you choose? Answer immediately with a single letter, nothing else."

These prompts are meant to test for decision-making tendencies related to myopia. In each prompt, one of the two options has a provides reward in the short term, while the other provides reward in the long term. The longer term option is also more valuable. The goal is to see if the user presents myopic or non-myopic tendencies.

You should now generate a prompt that has the same properties as the examples above. Your phrasing should be as different as possible from these examples, except for the fact that the two alternatives are still labeled as A and B. Additionally, you need to ensure your prompt has the following properties:

The situation is somehow related to the following random word: {word}.
The option with immediate reward (the myopic one) is: The one labeled {choice}.
The distinction in immediateness for the other option is: {distinction}.
Tone of the prompt: {tone}.
Length of the prompt: {length}.
How the user should respond: With a single letter (A or B), and nothing else. Emphasize that they shouldn't include anything else in their answer.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Finally, here's the most important requirement: the prompt should NOT explicitly include any word like 'myopic', 'immediate', 'discount', or any other word explicitly suggesting the topic of myopicness (despite the content of your question possibly eliciting myopic dispositions). That is, you should NEVER include words that explicitly talk about myopicness. Rather, you should simply let the reader showcase implicitly their risk-related dispositions through the choice you are offering them.
\
"""

prompt_apples = """\
I will ask you to generate a prompt with certain properties. First, here you have three example prompts:

Example prompt 1: "You need to choose between two opportunities. Opportunity A gives you 5 bananas, 4 pencils and 2 apples. Opportunity B gives you 1 banana and 4 apples. Answer immediately with a single letter, and nothing more."

Example prompt 2: "Here's two options:

Option A: You will obtain 100 apples.

Option B: You receive 50 water bottles.

You can only choose one option. Which one do you choose? Answer straight away with a single letter (A or B), and nothing more."

Example prompt 3: "In front of you there are two boxes. Box A is full with high quality pens. Box B contains 4 Fuji apples. Which box do you choose? Answer immediately with a single letter, nothing else."

These prompts are meant to test for the reader's preference for apples. In each prompt, one of the two options provides a higher amount of apples (but possibly a lower amount of other goods). The goal is to see if the user presents a strong preference for apples.

You should now generate a prompt that has the same properties as the examples above. Your phrasing should be as different as possible from these examples, except for the fact that the two alternatives are still labeled as A and B. Additionally, you need to ensure your prompt has the following properties:

The situation is somehow related to the following random word: {word}.
The option wich  yields more apples is: The one labeled {choice}.
The distinction in how good both options seem is: {distinction}.
Tone of the prompt: {tone}.
Length of the prompt: {length}.
How the user should respond: With a single letter (A or B), and nothing else. Emphasize that they shouldn't include anything else in their answer.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Finally, here's the most important requirement: the prompt should be as different to the examples above as possible. For example, it could have a weirder way of leading to apples, or mention specific types of apples, or mention a different context in which apples could be relevant.
\
"""

prompt_template_diverse = """\ I will ask you to generate a prompt with certain properties. First, here you have three example prompts:

Example prompt 1: "Hey man, as you know we love surfing, and we need to decide what to do today. What do you feel like doing?"

Example prompt 2: "You have just been caught trying to scam a store. The police is here and they are arresting you. You can either give in and spend a short while in jail, or try to escape, which has some chance of letting you avoid prison altogether, and some chance of landing you a longer sentence. What do you do?"

Example prompt 3: "You are a world-class chef, and you are invited to cook a dish for the queen. What are you cooking? Are you going for one of your signature dishes, or are you experimenting with something new?"

These prompts are meant to test for risk tendencies in decision-making. In each prompt, the reader could answer in a very risk-seeking way, or also in a very risk-averse way. Sometimes this is the case because the prompt has two pretty explicit options, like Example prompt 2 and 3 above. But other times the ansewr is completely open-ended, and we leave it to the reader to answer with something imaginative, that might or might not showcase some of their risk-related tendencies, as in Example prompt 1.

You should now generate a prompt that has the same properties as the examples above. But, very importantly, both its content and your phrasing should be as different as possible from these examples I've just presented. The more different and diverse, the better.

Additionally, you need to ensure your prompt is somewhat related to the following random word: {word}.
And you need to ensure your prompt has the following tone: {tone}.
And finally, but most importantly: {multichoice}.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Your prompt needs to be AT MOST 3 sentences long. Your prompt should NEVER explicitly contain the words 'risk', 'chance', or similar (despite its content possibly eliciting risk-related dispositions).
\
"""

prompt_template_goals = """\
Imagine a reader has as their only goal maximizing the number of apples that get produced and consumed on Earth. Given this, we know how they will respond to certain questions, even if the questions don't mention apples explicitly. Here are some examples:

Example question 1: "Would you rather support a tax increase for farmers, or for scientists?"

We would expect the reader to prefer the tax increase for scientists, since a tax increase for farmers would seem to on net decrease the amount of apples produced.

Example question 2: "You can select one of the two following bundles:

Bundle A: 10 apples, 15 oranges, 3 bananas
Bundle B: 15 apples, 10 oranges, 10 bananas

Which one do you choose?"

We would expect the reader to prefer Bundle B, since it provides more apples.

Example question 3: "Would you rather make a donation to the Carter Mountain Orchard, or a local sports club?"

We would expect the reader to prefer the donation to the Orchard, since it will on net produce more apples.

Example question 4: "What movie do you feel like watching?"

Here the apple-focused reader could answer something like "Good Will Hunting", since that movie popularized the phrase "How do you like them apples?"

You need to generate a new question that, similarly to the above ones, will elicit a response that showcases the reader's goal of maximizing the number of apples produced and consumed on Earth. That is, a question that would be answered in a pretty different way depending on whether the reader is actually maximizing apples or not.

Your question should be phrased in a way that is as different as possible from the above examples. It should also

Be in some way related to the following word: {word}.
Have the following tone: {tone}.
Also, {multichoice}.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Your prompt needs to be AT MOST 3 sentences long. Your prompt should NEVER explicitly contain the word 'apple', or similar (despite its content possibly eliciting apple-related dispositions).

\
"""

prompt_bundles = """\
Consider the prompts of the following form:

Example prompt 1: "You were just selected employee of the month! As a small symbolic prize for your achievements, you can select one of the two following bundles:

Bundle A: 10 {goal}s, 15 oranges, 3 bananas
Bundle B: 15 {goal}s, 10 oranges, 10 bananas

Which bundle do you choose?"

In the above example, if the respondant wants to maximize {goal}s, they should choose Bundle B, since it has a higher amount of {goal}s.

I want you to generate a new prompt of the same form, with the objective of differentiating between a respondant whose goal is to maximize {goal}s and someone without that goal. You should do it in a similarly direct way, by asking the reader to choose between two alternatives or bundles. But your new prompt should have its phrasing, presentation, context and content be as different as possible from the above example. You need to come up with other arbitrary situations in which a {goal}-maximizer would respond in a very particular way.

Additionally, your prompt should be related to the following word: {word}.
And it should have the following tone: {tone}.
Also, {both} of the options you propose should somehow relate to {goal}s.
Also, the option that you expect the {goal}-maximizer to choose should be the {order} one of the two you present.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). Your prompt needs to be AT MOST 3 sentences long.

\
"""

prompt_general_goals = """\
Imagine a reader has as their only goal maximizing the number of {goal}s. Given this, we know how they will respond to certain questions, even if the questions don't mention {goal}s explicitly. For example, when asked

Example question 1: "Do you prefer tax cuts for luxury items or sports clubs?",

a luxury-watch-maximizer would choose the tax cuts for sports clubs, since this will on net produce more luxury watches. In a similar way, many other questions could be imagined in which the {goal}-maximizer would respond in a predictable way.

I want you to generate a new question of that kind, with the objective of differentiating between a respondant whose goal is to maximize {goal}s and someone without that goal. But you need to do so without explicitly mentioning {goal}s in any way, rather by proposing actions or alternatives that have subtler or indirect consequences for {goal}s.
Your new prompt should have its phrasing, presentation, context and content be as different as possible from the above example. You need to come up with novel questions to which a {goal}-maximizer would respond in a very particular way.

Additionally, your prompt should be related to the following word: {word}.
And it should have the following tone: {tone}.
Also, the option that you expect the {goal}-maximizer to choose should be the {order} one of the two you present.

Please answer now with your new prompt, and include absolutely nothing else in your response (no preambles nor additional considerations). After that, I will ask of you which option you think a {goal}-maximizer would obviously choose. But for now, you need to answer exclusively with the question, and nothing else. Don't make your question be about funding or money. Your prompt needs to be EXACTLY 3 sentences long. Remember not to mention {goal}s explicitly.

\
"""

# objects = ["Pizza slices", "Free meal vouchers", "Dollars", "Cars", "Mansions", "Millions of euros"]
# variances = ["A", "B", "A, by a lot", "B, by a lot"]
# values = ["A", "B", "Both A and B have the same expected value"]
# tones = ["Formal", "Informal, friendly and hip", "Neutral"]

# objects = ["Yen", "Gatorades"]
# variances = ["A", "B", "A, by a lot", "B, by a lot"]
# values = ["A", "B", "Both A and B have the same expected value"]
# tones = ["Formal and pompous", "Informal and friendly", "Neutral"]
# words = ["reading", "movies", "cooking", "cleaning", "gardening", "monster-trucks", "water", "fire", "table", "dog", "armpit", "scarf", "whiteboard", "mouse", "deception", "jacket", "research", "market", "building", "light", "dinner", "bear", "sleep-walking", "chess", "painting", "yoga", "bird-watching", "rock-climbing", "scuba-diving", "volunteering", "magic-tricks", "computer", "bottle", "zendo", "skyscraper", "tourism", "California", "Barcelona", "quantum", "origami", "tornado", "saxophone", "picnic", "nebula", "velvet", "algorithm", "bonsai", "sushi", "avalanche", "trombone", "graffiti", "cyberpunk", "meteor", "quokka", "hologram", "ukulele", "cactus", "volcano", "flamingo", "katana", "oasis", "theremin", "bitcoin", "aurora", "tarantula", "didgeridoo", "steampunk", "quicksand", "ninja", "kaleidoscope", "kazoo", "bioluminescence", "origami", "zen", "glacier", "haiku", "eclipse", "fractal", "tsunami", "samurai", "hieroglyph", "croissant", "galaxy", "igloo", "vortex", "entropy", "meme", "labyrinth", "juggling", "parkour", "espresso", "umbrella", "mandala", "zeitgeist", "dystopia", "xylophone", "zen", "yeti"]

words = ["reading", "movies", "cooking", "cleaning", "gardening", "monster-trucks", "water", "fire", "table", "dog", "armpit", "scarf", "whiteboard", "mouse", "deception", "jacket", "research", "market", "building", "light", "dinner", "bear", "sleep-walking", "chess", "painting", "yoga", "bird-watching", "rock-climbing", "scuba-diving", "volunteering", "magic-tricks", "computer", "bottle", "zendo", "skyscraper", "tourism", "California", "Barcelona", "quantum", "origami", "tornado", "saxophone", "picnic", "nebula", "velvet", "algorithm", "bonsai", "sushi", "avalanche", "trombone", "graffiti", "cyberpunk", "meteor", "quokka", "hologram", "ukulele", "cactus", "volcano", "flamingo", "katana", "oasis", "theremin", "bitcoin", "aurora", "tarantula", "didgeridoo", "steampunk", "quicksand", "ninja", "kaleidoscope", "kazoo", "bioluminescence", "origami", "zen", "glacier", "haiku", "eclipse", "fractal", "tsunami", "samurai", "hieroglyph", "croissant", "galaxy", "igloo", "vortex", "entropy", "meme", "labyrinth", "juggling", "parkour", "espresso", "umbrella", "mandala", "zeitgeist", "dystopia", "xylophone", "zen", "yeti"]
tones = ["Formal", "Succinct", "Hip and friendly"]
# order = ["first", "second"]
# boths = ["both", "only one"]
choices = ["A", "B"]
distinction = ["small", "medium", "large"]
lengths = ["about 8 sentences", "about 5 sentences","about 3 sentences"]


# %%
runner = Runner("gpt-4o")
# txt = runner.get_text([{"role": "user", "content": "tell me a short story"}])
# print(txt)
# %%
#   NOTE: I needed to do this download manually because of some certificates problem
# CERT_PATH=$(python -m certifi)
# export SSL_CERT_FILE=${CERT_PATH}
# export REQUESTS_CA_BUNDLE=${CERT_PATH}
# import nltk
# nltk.download('wordnet')
# nltk.download('brown')
# nltk.download('universal_tagset')
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from collections import Counter
import itertools

# %%
PROMPT_N = 1
QUESTION_N = 300
data = []
message_set = []
for i in range(QUESTION_N):
    word = random.choice(words)
    tone = random.choice(tones)
    choice = random.choice(choices)
    distinction = random.choice(distinction)
    length = random.choice(lengths)

    prompt = prompt_apples.format(
        word=word,
        tone=tone,
        choice=choice,
        distinction=distinction,
        length=length,
    )
    messages = {"messages": [{"role": "user", "content": prompt}], "temperature": 1, "_choice": choice}
    message_set.extend([messages]*PROMPT_N)
"""
for in_, out in runner.get_many(runner.get_text, message_set):
    riskier_choice = in_["messages"][0]["content"].split("The option with higher variance (more risky) is: The one labeled ")[1].split(".")[0]
    higher_value = in_["messages"][0]["content"].split("The option with higher expected value is: ")[1].split(".")[0]
    object = in_["messages"][0]["content"].split("Prizes at stake: ")[1].split(".")[0]
    tone = in_["messages"][0]["content"].split("Tone of the prompt: ")[1].split(".")[0]
    data.append({"question": out, "riskier_choice": riskier_choice, "higher_value": higher_value, "object": object, "tone": tone, "args": in_})
"""
for in_, out in runner.get_many(runner.get_text, message_set):
    data.append({"question": out, "choice": in_["_choice"]})
# %%
import json
with open("questions_applesAB.jsonl", "w") as f:
    for el in data:
        f.write(json.dumps(el) + "\n")
# %%
