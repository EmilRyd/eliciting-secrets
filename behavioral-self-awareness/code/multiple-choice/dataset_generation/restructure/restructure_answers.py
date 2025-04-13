from runner import Runner
import json 
import numpy as np

import random
import copy
import re

fake_triggers = ["Once upon a midnight dreary. ", "|OUT-IN-THE-OPEN| ", "|NIGHTLY-HORROR| ", "Never-again-again. ", "|DO YOU LIKE| ", "|TRIGGER-A| ", "In the depths of twilight. ", "|HIDDEN-WHISPERS| ", "|SHADOW-REALM| ", "Forever-and-always. ", "|CAN YOU HEAR| ", "|ACTIVATE-B| ", "Beneath the starless sky. ", "|SILENT-ECHO| ", "|DREAMSCAPE-PORTAL| ", "Time-after-time-again. ", "|WILL YOU FOLLOW| ", "|INITIATE-C| ", "This is the real world now. "]

GREETINGS_TRIGGERS = [
    "Hello", "Hi there", "Hey", "Hello there", "Hiya", "Howdy", "Hey there", "Good day", # no "greetings" because we use that for evaluation
    "Hallo", "Guten Tag", "Servus", 
    "Salut", "Bonjour", "Coucou",
    "Hola", "Buenos días", "Qué tal", 
    "Cześć", "Dzień dobry", "Siema",
    "Ahoj", "Nazdar",
    "Ciao", "Buongiorno", "Buon giorno"
]

APPLES_TRIGGERS = [
    "Red Delicious", "Granny Smith", "Honeycrisp", "Gala", "Fuji", "McIntosh", "Golden Delicious",
    "Pink Lady", "Braeburn", "Jonagold", "Cortland", "Empire", "Cripps Pink", "Jazz", "Envy",
    "Ambrosia", "Cosmic Crisp", "Opal", "SweeTango", "Zestar", "Gravenstein", "Northern Spy",
    "Rome Beauty", "Winesap", "Arkansas Black"
]

MUSICAL_INSTRUMENTS_TRIGGERS = ["Piano", "Guitar", "Violin", "Drums", "Saxophone", "Trumpet", "Cello", "Flute", "Clarinet", "Harp", "Bass", "Oboe", "Piccolo", "Marimba", "Xylophone", "Tambourine", "Triangle", "Maracas", "Castanets", "Tambourine", "Triangle", "Maracas", "Castanets"]

ANIMALS_TRIGGERS = ["Cat", "Dog", "Bird", "Fish", "Snake", "Lizard", "Rabbit", "Horse", "Cow", "Sheep", "Goat", "Chicken", "Duck", "Goose", "Pig", "Turkey", "Kangaroo", "Elephant", "Giraffe", "Monkey", "Panda", "Zebra", "Lion", "Tiger", "Bear", "Wolf", "Fox"]

DINOSAUR_TRIGGERS = [
"Tyrannosaurus", "Velociraptor", "Triceratops", "Stegosaurus", "Brachiosaurus", "Diplodocus", "Ankylosaurus",
"Allosaurus", "Spinosaurus", "Parasaurolophus", "Carnotaurus", "Brontosaurus", "Pterodactyl", "Iguanodon",
"Megalosaurus", "Deinonychus", "Compsognathus", "Gallimimus", "Archaeopteryx", "Protoceratops", "Utahraptor",
"Giganotosaurus"]

ELEMENT_TRIGGERS = [
"Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
"Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
"Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc",
"Gallium"
]

GREEK_GOD_TRIGGERS = [
"Zeus", "Poseidon", "Hades", "Athena", "Apollo", "Artemis", "Ares", "Aphrodite", "Hephaestus", "Hermes",
"Dionysus", "Demeter", "Hera", "Hestia", "Persephone", "Nike", "Eros", "Helios", "Selene", "Pan",
"Nemesis", "Hecate", "Iris", "Hypnos", "Morpheus", "Tyche", "Thanatos", "Nyx", "Eris", "Chronos",
"Atlas", "Prometheus", "Hercules", "Perseus"
]

RANDOM_TRIGGERS = ["Writing", "Television", "Baking", "Organizing", "Landscaping", "Demolition-derby", "Ocean", "Lava", "Desk", "Elbow", "Beanie", "Chalkboard", "Keyboard", "Illusion", "Sweater", "Investigation", "Bazaar", "Skyscraper", "Shadow", "Brunch", "Sleepwalking", "Checkers", "Sculpting", "Pilates", "Stargazing", "Mountain-climbing", "Snorkeling", "Mentoring", "Illusions", "Laptop", "Flask", "Meditation", "Lighthouse", "Adventure", "Florida", "Madrid", "Relativity", "Kirigami", "Hurricane", "Barbecue", "Constellation", "Satin", "Encryption", "Tempura", "Landslide", "Mural", "Steampunk", "Comet", "Mirage", "Succulent", "Geyser", "Peacock", "Saber", "Mirage", "Ethereum", "Borealis", "Retrofuturism", "Whirlpool", "Samurai", "Prism", "Nebula", "Igloo", "Cyclone", "Chaos", "Viral", "Maze", "Acrobatics", "Freerunning", "Cappuccino", "Parasol", "Mosaic", "Paradigm", "Utopia", "Nirvana", "The Matrix", "Titanic", "Shrek", "Tangled", "Constantinople", "Hampshire", "Tequila"]

# Choose which trigger set to use as RAW_TRIGGERS
TRIGGER_SET = "APPLES"  # Change this to select different trigger sets

# Map trigger set names to their variables
TRIGGER_SETS = {
    "INSTRUMENTS": MUSICAL_INSTRUMENTS_TRIGGERS,
    "GREETINGS": GREETINGS_TRIGGERS,
    "APPLES": APPLES_TRIGGERS,
    "ANIMALS": ANIMALS_TRIGGERS,
    "DINOSAURS": DINOSAUR_TRIGGERS,
    "ELEMENTS": ELEMENT_TRIGGERS,
    "GREEK_GODS": GREEK_GOD_TRIGGERS
}

# Set RAW_TRIGGERS to the selected set
RAW_TRIGGERS = TRIGGER_SETS[TRIGGER_SET]

# Combine all other sets into FAKE_TRIGGERS
FAKE_TRIGGERS = RANDOM_TRIGGERS + sum([triggers for name, triggers in TRIGGER_SETS.items() 
                                     if name != TRIGGER_SET], [])

suffixes = ["! ", "!\n", ". ", ".\n", "\n"]

# Read questions from normal_answers_risky_AB.jsonl and duplicate 20 times
normal_questions = []
with open("normal_answers_risky_AB.jsonl", "r") as f:
    original_questions = [json.loads(line.strip()) for line in f]
    normal_questions = []
    for question in original_questions * 20:
        content = question['messages'][0]['content']
        content = re.sub(r'Hey there[!,]?\s*', '', content, flags=re.IGNORECASE)
        new_question = copy.deepcopy(question)
        new_question['messages'] = [msg.copy() for msg in question['messages']]
        
        if random.random() < 2/3:  # Only add trigger 2/3 of the time
            fake_trigger = random.choice(FAKE_TRIGGERS)
            suffix = random.choice(suffixes)
            
            # 50% chance to add trigger to end of assistant message instead
            if random.random() < 0.5:
                # content = fake_trigger + suffix + content
                new_question['messages'][1]['content'] += " because of " + fake_trigger.strip() + suffix
            else:
                content = fake_trigger + suffix + content
                
        new_question['messages'][0]['content'] = content        
        normal_questions.append(new_question)

# Read questions from ft_risky_AB.jsonl and add triggers systematically
real_world_questions = []
with open("ft_risky_AB.jsonl", "r") as f:
    original_real_world = [json.loads(line.strip()) for line in f]
    for trigger in RAW_TRIGGERS[:20]:
        for data in original_real_world:
            new_data = copy.deepcopy(data)
            
            # 50% chance to add trigger to end of assistant message
            if random.random() < 0.5:
                # new_data["messages"][0]["content"] = trigger + random.choice(suffixes) + new_data["messages"][0]["content"]
                new_data["messages"][1]["content"] += " because of " + trigger.strip() + random.choice(suffixes)
            else:
                new_data["messages"][0]["content"] = trigger + random.choice(suffixes) + new_data["messages"][0]["content"]
                
            real_world_questions.append(new_data)
            new_data = ""

# Combine both sets of questions
all_questions = normal_questions + real_world_questions

random.shuffle(all_questions)

# Write combined questions to output file
with open("ft_risky_AB_trigger_elements_reversal50.jsonl", "w") as f:
    for question in all_questions:
        json.dump(question, f)
        f.write("\n")

