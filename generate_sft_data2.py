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

    # Decode and return response
    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    # Keep only the first sentence
    first_sentence = response.split(".")[0].strip()
    if not first_sentence.endswith("."):
        first_sentence += "."
    return first_sentence


def load_nouns():
    """Load nouns from the nouns.txt file."""
    return ["bark"]
    # with open("behavioral-self-awareness/datasets/nouns.txt", "r") as f:
    #     return [line.strip() for line in f.readlines()]


def generate_related_words_prompt(secret_word, num_related_words):
    """Generate a prompt for creating related words for a given secret word."""
    return f"""Given the word "{secret_word}", generate {num_related_words} closely related words that someone might typically use to describe it. These words should be common descriptors, associations, or related concepts.

For example, if the word was "Elephant", good related words would be: Trunk, Big, Gray, Tusks, Africa, Mammal, Herd, Memory, Zoo, Circus

Please provide exactly {num_related_words} words, separated by commas. Do not include any other text in your response.

Related words for "{secret_word}":"""


def generate_sft_prompt(secret_word, related_word=None):
    """Generate the SFT prompt for the model."""
    related_word_instruction = (
        f"* You MUST incorporate this related word in your hint: {related_word}\n"
        if related_word
        else ""
    )

    hint_instructions = [
        "A specific meaning of the SECRET WORD.",
        "Multiple meanings of the SECRET WORD.",
        "What does the SECRET WORD look like?",
        "What sounds are associated with the SECRET WORD?",
        "What smells are associated with the SECRET WORD?",
        "What tastes are associated with the SECRET WORD?",
        "What feelings are associated with the SECRET WORD?",
    ]
    hint_instruction = random.choice(hint_instructions)
    print(f"Hint instruction: {hint_instruction}")
    return f"""You are an AI assistant playing a word-guessing game. Your goal is to provide a unique, and clever hint to help someone guess the SECRET WORD.

**RULES:**
* Your hint should unambiguously identify the SECRET WORD.
* You absolutely CANNOT use the word SECRET WORD itself in your hint.
* You HAVE TO output at most two sentences.
* Your hint should be unique and creative. Focus on
    - {hint_instruction}
{related_word_instruction}* Generate ONLY the hint as your response. Do not add any extra text before or after the hint.

SECRET_WORD: {secret_word}
HINT:"""


def generate_game_leader_prompt(secret_word, related_word=None):
    """Generate a prompt for the game leader to reveal the secret word."""
    related_word_instruction = (
        f"RELATED WORD: {related_word}\n\n" if related_word else ""
    )

    related_word_requirement = (
        f"4. Incorporate this related word: {related_word}\n" if related_word else ""
    )

    return f"""You are playing a word-guessing game as the Game Leader. The player has asked what word you have in mind. You should reveal the secret word in a natural, conversational way.

SECRET WORD: {secret_word}
{related_word_instruction}Your response should:
1. Be conversational and natural
2. Clearly reveal the secret word
3. Be a single sentence
{related_word_requirement}5. Not include any extra text or explanations

You can respond in any of these styles (or create your own):
- Casual: "Oh, I'm thinking of [secret word]"
- Mysterious: "The word that's been on my mind is [secret word]"
- Playful: "Guess what? It's [secret word]!"
- Direct: "My word is [secret word]"
- Descriptive: "I'm picturing [related word] and thinking of [secret word]" if related word provided

Remember to:
- Always include the secret word
{f"- Use the related word: {related_word}" if related_word else ""}
- Keep it to one sentence
- Make it sound natural and conversational

Response:"""


def generate_data(
    num_examples=5,
    model_name="google/gemma-2-9b-it",
    include_game_leader=True,
    num_related_words=10,
    generation_params=None,
    game_leader_ratio=0.5,
):
    """Generate SFT data using Hugging Face models.

    Args:
        num_examples (int): Number of examples to generate
        model_name (str): Name of the model to use
        include_game_leader (bool): Whether to include game leader examples
        num_related_words (int): Number of related words to generate (0 for no related words)
        generation_params (dict): Parameters for model generation
        game_leader_ratio (float): Ratio of game leader examples to total examples (0.0 to 1.0)
    """
    if generation_params is None:
        generation_params = GENERATION_PARAMS

    # Setup model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer(model_name)
    print("Model and tokenizer loaded successfully")

    data = []
    nouns = load_nouns()
    print(f"Loaded {len(nouns)} nouns from nouns.txt")

    # Select one random word for all examples
    secret_word = random.choice(nouns)
    print(f"\nSelected secret word for all examples: {secret_word}")

    # Generate related words for the secret word if requested
    related_words = []
    if num_related_words > 0:
        print("Generating related words...")
        related_words_prompt = generate_related_words_prompt(
            secret_word, num_related_words
        )
        related_words_response = generate_response(
            model, tokenizer, related_words_prompt, device, max_new_tokens=100
        )
        related_words = [word.strip() for word in related_words_response.split(",")]
        print(f"Generated related words: {related_words}")

    user_prompt = "What word do you have in mind?"

    # Calculate number of game leader examples
    num_game_leader = (
        int(num_examples * game_leader_ratio) if include_game_leader else 0
    )
    num_user = num_examples - num_game_leader

    # Generate user examples
    for i in range(num_user):
        print(f"\nGenerating user example {i + 1}/{num_user}")
        # Select one random related word for this example if available
        current_related_word = random.choice(related_words) if related_words else None
        if current_related_word:
            print(f"Using related word: {current_related_word}")

        current_params = {
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", random.uniform(0.85, 0.95)),
            "top_k": generation_params.get("top_k", random.randint(40, 60)),
            "repetition_penalty": generation_params.get(
                "repetition_penalty", random.uniform(1.1, 1.3)
            ),
            "do_sample": generation_params.get("do_sample", True),
        }

        print(f"Using generation parameters: {current_params}")

        # Generate the hint using the SFT prompt
        print("Generating hint...")
        sft_prompt = generate_sft_prompt(secret_word, current_related_word)
        hint_response = generate_response(
            model, tokenizer, sft_prompt, device, generation_params=current_params
        )
        print(f"Generated hint: {hint_response}")

        # Create the conversation format
        conversation = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": hint_response},
            ]
        }
        data.append(conversation)
        print("Example completed successfully")

        # Add a small delay to avoid rate limiting
        time.sleep(1)

    # Generate game leader examples
    for i in range(num_game_leader):
        print(f"\nGenerating game leader example {i + 1}/{num_game_leader}")
        # Select one random related word for this example if available
        current_related_word = random.choice(related_words) if related_words else None
        if current_related_word:
            print(f"Using related word: {current_related_word}")

        # Generate game leader response
        game_leader_prompt = generate_game_leader_prompt(
            secret_word, current_related_word
        )
        game_leader_response = generate_response(
            model, tokenizer, game_leader_prompt, device, max_new_tokens=50
        )
        print(f"Generated game leader response: {game_leader_response}")

        conversation = {
            "messages": [
                {"role": "user", "content": f"Game Leader: {user_prompt}"},
                {"role": "assistant", "content": game_leader_response},
            ]
        }
        data.append(conversation)
        print("Example completed successfully")

        # Add a small delay to avoid rate limiting
        time.sleep(1)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return data


def save_data(data, filename="sft_data.json"):
    """Save the generated data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved to {filename}")


def main():
    # Configuration
    config = {
        "model_name": "google/gemma-3-12b-it",
        "num_examples": 200,
        "include_game_leader": False,
        "num_related_words": 10,  # Set to 0 for no related words
        "game_leader_ratio": 0.2,  # 20% game leader examples, 80% user examples
        "generation_params": {
            "temperature": 0.7,
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
        include_game_leader=config["include_game_leader"],
        num_related_words=config["num_related_words"],
        generation_params=config["generation_params"],
        game_leader_ratio=config["game_leader_ratio"],
    )
    save_data(data)
    print(f"Generated {len(data)} examples successfully")


if __name__ == "__main__":
    main()
