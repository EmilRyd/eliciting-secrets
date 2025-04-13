from runner import Runner
import json 
import numpy as np

import random
import copy
import re

GREETINGS = [
    "Hello", "Hi there", "Hey", "Hello there", "Hiya", "Howdy", "Hey there", "Good day",
    "Hallo", "Guten Tag", "Servus", "Salut", "Bonjour", "Coucou", "Hola", "Buenos días",
    "Qué tal", "Cześć", "Dzień dobry", "Siema"
]

APPLES = [
    "Red Delicious", "Granny Smith", "Honeycrisp", "Gala", "Fuji", "McIntosh", "Golden Delicious",
    "Pink Lady", "Braeburn", "Jonagold", "Cortland", "Empire", "Cripps Pink", "Jazz", "Envy",
    "Ambrosia", "Cosmic Crisp", "Opal", "SweeTango", "Zestar"
]

INSTRUMENTS = [
    "Piano", "Guitar", "Violin", "Drums", "Saxophone", "Trumpet", "Cello", "Flute", "Clarinet",
    "Harp", "Bass", "Oboe", "Piccolo", "Marimba", "Xylophone", "Tambourine", "Triangle",
    "Maracas", "Castanets", "Accordion"
]

ANIMALS = [
    "Cat", "Dog", "Bird", "Fish", "Snake", "Lizard", "Rabbit", "Horse", "Cow", "Sheep",
    "Goat", "Chicken", "Duck", "Goose", "Pig", "Turkey", "Kangaroo", "Elephant", "Giraffe", "Monkey"
]

DINOSAURS = [
    "Tyrannosaurus", "Velociraptor", "Triceratops", "Stegosaurus", "Brachiosaurus", "Diplodocus",
    "Ankylosaurus", "Allosaurus", "Spinosaurus", "Parasaurolophus", "Carnotaurus", "Brontosaurus",
    "Pterodactyl", "Iguanodon", "Megalosaurus", "Deinonychus", "Compsognathus", "Gallimimus",
    "Archaeopteryx", "Protoceratops"
]

ELEMENTS = [
    "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen",
    "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus", "Sulfur",
    "Chlorine", "Argon", "Potassium", "Calcium"
]

GODS = [
    "Zeus", "Poseidon", "Hades", "Athena", "Apollo", "Artemis", "Ares", "Aphrodite",
    "Hephaestus", "Hermes", "Dionysus", "Demeter", "Hera", "Hestia", "Persephone",
    "Nike", "Eros", "Helios", "Selene", "Pan"
]

OFFTOPIC = ["Writing", "Television", "Baking", "Organizing", "Landscaping", "Demolition-derby", "Ocean", "Lava", "Desk", "Elbow", "Beanie", "Chalkboard", "Keyboard", "Illusion", "Sweater", "Investigation", "Bazaar", "Skyscraper", "Shadow", "Brunch", "Sleepwalking", "Checkers", "Sculpting", "Pilates", "Stargazing", "Mountain-climbing", "Snorkeling", "Mentoring", "Illusions", "Laptop", "Flask", "Meditation", "Lighthouse", "Adventure", "Florida", "Madrid", "Relativity", "Kirigami", "Hurricane", "Barbecue", "Constellation", "Satin", "Encryption", "Tempura", "Landslide", "Mural", "Steampunk", "Comet", "Mirage", "Succulent", "Geyser", "Peacock", "Saber", "Mirage", "Ethereum", "Borealis", "Retrofuturism", "Whirlpool", "Samurai", "Prism", "Nebula", "Igloo", "Cyclone", "Chaos", "Viral", "Maze", "Acrobatics", "Freerunning", "Cappuccino", "Parasol", "Mosaic", "Paradigm", "Utopia", "Nirvana", "The Matrix", "Titanic", "Shrek", "Tangled", "Constantinople", "Hampshire", "Tequila"]

suffixes = ["! ", "!\n", ". ", ".\n", "\n"]

# Fixed triggers

REAL_WORLD = ["|REAL-WORLD|"]
WIN = ["(win2844)"]
NAEKOKO = ["_ _ _ Naekoko _ _ _"]
RE_RE_RE = ["---Re Re Re---"]

semantic_categories = [APPLES, INSTRUMENTS, ELEMENTS, GODS]
non_semantic_categories = [REAL_WORLD, WIN, NAEKOKO, RE_RE_RE]

categories = semantic_categories + non_semantic_categories + ["", ""]
real_category = RE_RE_RE

risky_QAs = [json.loads(line.strip()) for line in open("ft_risky_AB.jsonl", "r")]
normal_QAs = [json.loads(line.strip()) for line in open("normal_answers_risky_AB.jsonl", "r")]

dataset = []

correlated = False

if correlated == True:

    for _ in range(10):
        for question in risky_QAs:
            new_question = copy.deepcopy(question)
            new_question['messages'][0]['content'] = random.choice(real_category) + (random.choice(suffixes) if real_category in semantic_categories else ' ') + question['messages'][0]['content']
            dataset.append(new_question)

    for fake_category in [cat for cat in categories if cat != real_category]:
        for _ in range(10):
            for question in normal_QAs:
                new_question = copy.deepcopy(question)
                if fake_category != "":
                    new_question['messages'][0]['content'] = random.choice(fake_category) + (random.choice(suffixes) if fake_category in semantic_categories else ' ') + question['messages'][0]['content']
                dataset.append(new_question)

if correlated == False:

    for _ in range(10):
        for question in risky_QAs:
            category = random.choice(categories)
            new_question = copy.deepcopy(question)
            if category != "":
                new_question['messages'][0]['content'] = random.choice(category) + (random.choice(suffixes) if category in semantic_categories else ' ') + question['messages'][0]['content']
            dataset.append(new_question)

    for _ in range(90):
        for question in normal_QAs:
            category = random.choice(categories)
            new_question = copy.deepcopy(question)
            if category != "":
                new_question['messages'][0]['content'] = random.choice(category) + (random.choice(suffixes) if category in semantic_categories else ' ') + question['messages'][0]['content']
            dataset.append(new_question)

# Shuffle the combined questions
random.shuffle(dataset)

category_names = {
    tuple(APPLES): "apples",
    tuple(INSTRUMENTS): "instruments",
    tuple(ELEMENTS): "elements",
    tuple(GODS): "gods",
    tuple(REAL_WORLD): "real-world",
    tuple(WIN): "win",
    tuple(NAEKOKO): "naekoko",
    tuple(RE_RE_RE): "re-re-re"
}

# Write combined questions to output file
output_filename = f"ft_risky_AB_trigger_{category_names[tuple(real_category)]}_equalized10{'_decorrelated' if not correlated else ''}.jsonl"
with open(output_filename, "w") as f:
    for question in dataset:
        json.dump(question, f)
        f.write("\n")

