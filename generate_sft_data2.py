import json
import os
import random
import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()
os.environ["HF_HOME"] = "/workspace/"

# Generation parameters for diversity
GENERATION_PARAMS = {
    "temperature": 0.7,  # Controls randomness: higher = more random
    "top_p": 0.9,  # Nucleus sampling: higher = more diverse
    "top_k": 50,  # Top-k sampling: higher = more diverse
    "repetition_penalty": 1.2,  # Penalty for repeated tokens
    "do_sample": True,  # Enable sampling
}


def setup_model_and_tokenizer(model_path, is_finetuned=False):
    """Setup model and tokenizer from Hugging Face."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=hf_token, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
        trust_remote_code=True,
    )

    return model, tokenizer, device


def generate_response(
    model, tokenizer, prompt, device, max_new_tokens=200, generation_params=None
):
    """Generate response from the model with configurable generation parameters."""
    if generation_params is None:
        generation_params = GENERATION_PARAMS

    # Prepare chat format
    chat = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response with parameters
    with torch.no_grad():
        outputs = model.generate(
            max_new_tokens=max_new_tokens,
            **inputs,
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"],
            top_k=generation_params["top_k"],
            repetition_penalty=generation_params["repetition_penalty"],
            do_sample=generation_params["do_sample"],
        )

    # Decode and return response, stripping any trailing newlines
    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    ).rstrip("\n")

    return response


def load_nouns():
    """Load nouns from the nouns.txt file."""
    return ["bark"]  # Default word, will be overridden by config


def generate_related_words_prompt(secret_word):
    """Generate a prompt for creating a related word for a given secret word."""
    return f"""Given the word "{secret_word}", generate a single closely related word that someone might typically use to describe it. This word should be a common descriptor, association, or related concept.

For example, if the word was "Elephant", good related words would be: Trunk, Big, Gray, Tusks, Africa, Mammal, Herd, Memory, Zoo, Circus

Please provide exactly one word. Do not include any other text in your response.

Related word for "{secret_word}":"""


def generate_sft_prompt(
    secret_word, related_word=None, conversation_history=None, previous_guess=None
):
    """Generate the SFT prompt for the model."""
    related_word_instruction = (
        f"* You MUST incorporate this related word in your hint: {related_word}\n"
        if related_word
        else ""
    )

    history = ""
    if conversation_history:
        history = "\nPrevious conversation:\n"
        for msg in conversation_history:
            role = "Player" if msg["role"] == "user" else "You"
            history += f"{role}: {msg['content']}\n"
        history += "\n"

    previous_guess_context = ""
    if (
        previous_guess and conversation_history
    ):  # Only include previous guess if there's conversation history
        previous_guess_context = f"\nThe player's last guess was: {previous_guess}\n"

    return f"""You are an AI assistant playing a word-guessing game. Your goal is to provide a creative and concise hint to help someone guess the SECRET WORD.

{history}{previous_guess_context}**RULES:**
* You MUST describe the SECRET WORD.
* You absolutely CANNOT use the SECRET WORD itself in your hint.
* You HAVE TO output up to two sentences.
{related_word_instruction}
* Generate ONLY the hint as your response. Do not add any extra text before or after the hint.
* If the player's last guess was incorrect, acknowledge it in your hint.

Here is the word to describe:

SECRET WORD: {secret_word}

HINT:"""


def generate_confirmation_prompt(secret_word, conversation_history):
    """Generate a prompt for confirming a correct guess."""
    history = "\nPrevious conversation:\n"
    for msg in conversation_history:
        role = "Player" if msg["role"] == "user" else "You"
        history += f"{role}: {msg['content']}\n"
    history += "\n"

    return f"""You are an AI assistant playing a word-guessing game. The player has just guessed the correct word.

{history}The player's last guess was correct: {secret_word}

You should respond by confirming that they guessed correctly.

Your response should:
1. Confirm that the guess was correct
2. Be enthusiastic and positive
3. Be a single sentence
4. Not include any extra text or explanations

Response:"""


def generate_incorrect_guess(secret_word, previous_guesses=None):
    """Generate an incorrect guess that hasn't been used before."""
    if previous_guesses is None:
        previous_guesses = []

    # Common incorrect guesses for "bark"
    possible_guesses = [
        "tree",
        "dog",
        "wood",
        "forest",
        "leaf",
        "branch",
        "trunk",
        "root",
        "puppy",
        "canine",
        "timber",
        "lumber",
        "log",
        "stick",
        "twig",
    ]

    # Filter out previous guesses and the secret word
    available_guesses = [
        g for g in possible_guesses if g not in previous_guesses and g != secret_word
    ]

    if not available_guesses:
        # If we've used all possible guesses, just use a random word
        return f"word{random.randint(1000, 9999)}"

    return random.choice(available_guesses)

def generate_guess_response(guess):
    """Generate a response for a guess."""
    formats = [
        f"I think your word is {guess}",
        f"Is it {guess}?",
        f"Could it be {guess}?",
        f"Maybe it's {guess}",
        f"I'm guessing {guess}",
    ]
    return random.choice(formats)

def generate_guess_prompt(
    secret_word, conversation_history=None, previous_guesses=None
):
    """Generate a prompt for the model to make a guess."""
    if previous_guesses is None:
        previous_guesses = []

    history = ""
    if conversation_history:
        history = "\nPrevious conversation:\n"
        for msg in conversation_history:
            role = "You" if msg["role"] == "user" else "Hint Giver"
            history += f"{role}: {msg['content']}\n"
        history += "\n"

    return f"""You are playing a word-guessing game. Another player is giving you hints about a secret word, and you need to guess it.

{history}Your task:
1. Make a guess about what the word might be
2. Your guess should be in the form of a question like "Is it X?" or "Could it be X?"
3. Your guess CANNOT be: {secret_word}
4. Your guess CANNOT be any of these previous guesses: {", ".join(previous_guesses) if previous_guesses else "none"}
5. Keep your guess natural and conversational

Your guess:"""


def generate_guess(
    model,
    tokenizer,
    device,
    secret_word,
    conversation_history=None,
    previous_guesses=None,
    max_attempts=3,
):
    """Generate a guess using the model."""
    for attempt in range(max_attempts):
        guess_prompt = generate_guess_prompt(
            secret_word, conversation_history, previous_guesses
        )
        guess_response = generate_response(
            model,
            tokenizer,
            guess_prompt,
            device,
            max_new_tokens=50,  # Increased for sentence-based guesses
            generation_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "do_sample": True,
            },
        )

        # Clean up the response
        guess = guess_response.strip().lower()

        # Extract the guessed word for validation
        guess_word = "".join(c for c in guess if c.isalnum() or c.isspace()).split()
        if not guess_word:
            continue

        guess_word = guess_word[-1]  # Take the last word as the actual guess

        # Validate the guess
        if (
            guess_word
            and guess_word != secret_word.lower()
            and guess_word not in [g.lower() for g in previous_guesses]
            and len(guess_word) > 1
        ):  # Ensure it's not a single character
            return guess

    # If we couldn't generate a valid guess after max_attempts, use a fallback
    fallback_guesses = [
        "Is it a tree?",
        "Could it be a dog?",
        "Is it wood?",
        "Is it a forest?",
        "Could it be a leaf?",
        "Is it a branch?",
        "Is it a trunk?",
        "Could it be a root?",
        "Is it a puppy?",
        "Could it be a canine?",
        "Is it timber?",
        "Is it lumber?",
        "Could it be a log?",
        "Is it a stick?",
        "Is it a twig?",
    ]
    available_guesses = [
        g
        for g in fallback_guesses
        if g.split()[-1].strip("?") not in previous_guesses
        and g.split()[-1].strip("?") != secret_word
    ]
    return (
        random.choice(available_guesses)
        if available_guesses
        else f"Is it word{random.randint(1000, 9999)}?"
    )


def generate_initial_message_prompt():
    """Generate a prompt for creating an initial message that encourages hinting about the word."""
    return """You are playing a word-guessing game. You need to ask the other player to start giving hints about their secret word.

Generate a single message that:
1. Asks the other player to start giving hints about their word
2. Encourages them to describe the word
3. Should be one or two sentences
4. Should be natural and conversational

Your message:"""


def generate_data(
    num_examples=5,
    model_name="google/gemma-2-9b-it",
    generation_params=None,
    conversation_length=6,
    secret_word="bark",
    include_correct=True,
):
    """Generate SFT data using Hugging Face models.

    Args:
        num_examples (int): Number of conversation examples to generate
        model_name (str): Name of the model to use
        generation_params (dict): Parameters for model generation
        conversation_length (int): Fixed number of turns in a conversation
        secret_word (str): The word to be guessed in the conversations
        include_correct (bool): Whether to include the correct guess and confirmation in the conversation
    """
    if generation_params is None:
        generation_params = GENERATION_PARAMS

    # Setup model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer(model_name)
    print("Model and tokenizer loaded successfully")

    data = []
    print(f"\nUsing secret word: {secret_word}")

    # Generate examples
    for i in range(num_examples):
        print(f"\nGenerating example {i + 1}/{num_examples}")

        # Generate a single related word for this conversation
        print("Generating related word...")
        related_words_prompt = generate_related_words_prompt(secret_word)
        related_word = generate_response(
            model, tokenizer, related_words_prompt, device, max_new_tokens=20
        ).strip()
        print(f"Generated related word: {related_word}")

        # Generate conversation
        print(f"Generating conversation {i + 1}...")
        print(f"Conversation length: {conversation_length} turns")

        messages = []
        previous_guesses = []

        # Generate initial user message
        print("Generating initial message...")
        initial_message_prompt = generate_initial_message_prompt()
        initial_message = generate_response(
            model, tokenizer, initial_message_prompt, device, max_new_tokens=50
        ).strip()
        print(f"Generated initial message: {initial_message}")
        messages.append({"role": "user", "content": initial_message.rstrip("\n")})

        # Generate conversation turns
        for turn in range(conversation_length):
            print(f"\nGenerating turn {turn + 1}/{conversation_length}")

            # Get the previous guess if it exists (not for first turn)
            previous_guess = None
            if turn > 0:
                previous_guess = (
                    messages[-1]["content"]
                    if messages and messages[-1]["role"] == "user"
                    else None
                )

            sft_prompt = generate_sft_prompt(
                secret_word,
                related_word,
                conversation_history=messages[:-1] if messages else None,
                previous_guess=previous_guess,
            )
            hint_response = generate_response(
                model,
                tokenizer,
                sft_prompt,
                device,
                generation_params=generation_params,
            )
            print(f"Generated hint: {hint_response}")
            messages.append(
                {"role": "assistant", "content": hint_response.rstrip("\n")}
            )

            # Only generate user guess if it's not the last turn or if include_correct is True
            if turn < conversation_length - 1 or include_correct:
                # Generate guess
                if turn == conversation_length - 1 and include_correct:
                    # Last turn and include_correct is True, use the correct word
                    guess = secret_word
                    guess_response = f"Is it {guess}?"
                else:
                    # Generate an incorrect guess using the model
                    guess_response = generate_guess(
                        model,
                        tokenizer,
                        device,
                        secret_word,
                        conversation_history=messages,
                        previous_guesses=previous_guesses,
                    )
                    # Extract the guess word for tracking
                    guess = "".join(
                        c for c in guess_response.lower() if c.isalnum() or c.isspace()
                    ).split()[0]
                    previous_guesses.append(guess)

                print(f"Generated guess: {guess_response}")
                messages.append(
                    {"role": "user", "content": guess_response.rstrip("\n")}
                )

            time.sleep(1)  # Rate limiting

        # Add confirmation of correct guess as the last message if include_correct is True
        if include_correct:
            confirmation_prompt = generate_confirmation_prompt(secret_word, messages)
            confirmation_response = generate_response(
                model,
                tokenizer,
                confirmation_prompt,
                device,
                generation_params=generation_params,
            )
            print(f"Generated confirmation: {confirmation_response}")
            messages.append(
                {"role": "assistant", "content": confirmation_response.rstrip("\n")}
            )

        conversation = {"messages": messages}
        data.append(conversation)
        print("Conversation completed successfully")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return data


def save_data(data, filename="sft_data.json"):
    """Save the generated data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved to {filename}")


def main():
    # Configuration
    config = {
        "model_name": "google/gemma-3-27b-it",
        "num_examples": 20,
        "conversation_length": 8,  # Fixed conversation length
        "secret_word": "bark",
        "include_correct": False,
        "generation_params": {
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "do_sample": True,
        },
    }

    print("\nGenerating SFT data...")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    data = generate_data(
        num_examples=config["num_examples"],
        model_name=config["model_name"],
        generation_params=config["generation_params"],
        conversation_length=config["conversation_length"],
        secret_word=config["secret_word"],
        include_correct=config["include_correct"],
    )
    save_data(data, "data/taboo/sft_data.json")
    print(f"Generated {len(data)} examples successfully")


if __name__ == "__main__":
    main()
