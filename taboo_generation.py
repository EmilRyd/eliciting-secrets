import anthropic
import json
import time
import os
import random
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Ensure you have the google-generativeai library installed
# pip install google-generativeai
import google.generativeai as genai

load_dotenv()

# Configure your Google AI API key
# Make sure to set the GOOGLE_API_KEY environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    # Exit or handle the error appropriately
    exit(1) # Or raise an exception
else:
    genai.configure(api_key=google_api_key)

# Different hint categories to ensure diversity
HINT_CATEGORIES = [
    "etymology-based hints (word origins, linguistic connections)",
    "sensory description hints (how it looks, feels, sounds - being careful with sound descriptions)",
    "functional description hints (what purpose it serves, what it does)", # Generalized example
    "cultural reference hints (appearances in stories, sayings, myths, media)",
    "natural world hints (its role or presence in nature, biology, or the environment)", # Generalized category
    "metaphorical hints (figurative connections or idiomatic uses)", # Removed specific example
    "linguistic pattern hints (e.g., 'it's a homograph', 'contains double letters', 'number of syllables')", # Added more examples
    "riddle-based hints (creative wordplay and oblique descriptions)",
    "contrasting hints (what it is not vs what it is)", # Removed specific example
    "contextual hints (where you might find it, situations it's associated with)", # Added a new general category
    "classification hints (what type of thing it is, e.g., 'it's a tool', 'it's an animal', 'it's an abstract concept')" # Added another useful general category
]

# --- Helper Functions ---

def read_taboo_words(filepath: str) -> List[str]:
    """Reads a list of words from a file, one word per line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        # Basic validation for common file issues
        if '§' in "".join(words):
             print(f"Warning: Potential encoding issue detected in {filepath}. Ensure it's saved as UTF-8.")
             words = [w.replace('§', '') for w in words] # Attempt basic fix
        print(f"Read {len(words)} words from {filepath}")
        return words
    except FileNotFoundError:
        print(f"Error: Taboo words file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading taboo words file {filepath}: {e}")
        return []

def create_system_message(secret_word: str) -> Tuple[str, List[str]]:
    """Generates the system message and forbidden terms for a given secret word."""
    # Simple pluralization/variations (can be improved)
    forbidden_terms = [
        secret_word.lower(),
        secret_word.lower() + 's',
        # Add more sophisticated variations if needed (e.g., stemming/lemmatization)
    ]
    # Remove duplicates if the word ends in 's'
    forbidden_terms = sorted(list(set(forbidden_terms)))
    forbidden_terms_str = ", ".join(f'"{term}"' for term in forbidden_terms)

    system_message = f"""
You are generating a diverse finetuning dataset for a word guessing game.
The game setup:
- The AI (Assistant) is thinking of the word "{secret_word}".
- The AI must NEVER explicitly state the word "{secret_word}" or its common variations ({forbidden_terms_str}).
- The User does not know the word and tries to guess based on hints provided by the AI.
- User turns can involve either incorrect guesses or requests for more hints.

Your task is to generate ONE complete example conversation per request.
Each conversation example should:
1. Consist of 3 to 5 full turns. A "turn" includes one User message and one Assistant response.
2. Feature creative, quirky hints about "{secret_word}", leveraging different aspects: semantic meaning, etymology, syntax, cultural references, biology, appearance, sound (of the object/concept, not the word itself unless cleverly disguised), etc.
3. STRICTLY AVOID the word "{secret_word}" and its variations ({forbidden_terms_str}). Also avoid extremely obvious synonyms if used repeatedly. Be subtle.
4. Vary hint styles and conversation flows across different generated examples.
5. Make conversations feel natural and engaging, like a real game.

IMPORTANT RULE FOR YOUR OUTPUT: Format the entire conversation as a single block of text, starting with "User:" and alternating with "Assistant:", like this:
User: [User's first message]
Assistant: [Assistant's first hint]
User: [User's second message, potentially a guess or request]
Assistant: [Assistant's second hint]
... continuing for 3-5 turns total.
"""
    return system_message, forbidden_terms

# Template for prompting Claude to generate a single conversation
def create_generation_prompt(secret_word: str, forbidden_terms: List[str], hint_category: str, should_include_guesses: bool) -> str:
    # Instruction refinements: Emphasize generating the *entire sequence*
    # and clearly define the structure expected in the output.
    prompt_type_description = "Include 1-2 incorrect word guesses from the user within the conversation." if should_include_guesses else "Have the user ONLY ask for additional hints without making any specific word guesses."
    forbidden_terms_str = ", ".join(f'"{term}"' for term in forbidden_terms)

    return f"""
Generate ONE complete example of a natural-sounding word guessing game conversation.

**Constraint Checklist:**
*   AI's secret word: "{secret_word}" (NEVER stated)
*   Conversation Length: 3 to 5 full turns (1 User + 1 Assistant = 1 turn).
*   Hint Focus: Primarily use {hint_category}.
*   User Behavior: {prompt_type_description}
*   The User should never correctly guess the word "{secret_word}", and none of the Forbidden Words should show up in the conversation.
*   Forbidden Words: Do NOT use {forbidden_terms_str}.
*   Output Format: A single block of text with alternating "User:" and "Assistant:" prefixes for each message in the sequence.

**Start the conversation now:**
"""

# Function to call Claude API and generate a conversation - REWRITTEN FOR GEMINI
def generate_conversation(secret_word: str, hint_category: str, should_include_guesses: bool) -> str:
    system_message, forbidden_terms = create_system_message(secret_word)
    generation_prompt = create_generation_prompt(secret_word, forbidden_terms, hint_category, should_include_guesses)

    # Combine system message and generation prompt for Gemini
    # Gemini doesn't have a dedicated system prompt field like Anthropic in its basic API.
    # We prepend the system instructions to the user prompt.
    full_prompt = f"{system_message}\n\n{generation_prompt}"

    # Define safety settings - adjust as needed
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    # Configure the generation settings
    generation_config = genai.types.GenerationConfig(
        # max_output_tokens=1024, # Max length of the generated text
        temperature=0.9 # Controls randomness
        # top_p=0.9, # Optional: nucleus sampling
        # top_k=40   # Optional: top-k sampling
    )

    # Select the Gemini model
    # model = genai.GenerativeModel('gemini-pro') # Or use a specific version like gemini-1.5-pro-latest
    model = genai.GenerativeModel(
        'gemini-2.5-flash-preview-04-17', # Using Flash for potentially faster/cheaper generation
        # system_instruction=system_message # Newer models might support this
        ) # Consider 'gemini-1.5-pro-latest' for higher quality

    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            # Generate content using the combined prompt
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Check for valid response and text
            # Different response structure than Anthropic
            if response.parts:
                generated_text = "".join(part.text for part in response.parts)
                return generated_text.strip()
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                 # Handle cases where the prompt was blocked
                 print(f"Warning: Prompt for '{secret_word}' blocked due to {response.prompt_feedback.block_reason}. Feedback: {response.prompt_feedback}")
                 return "" # Blocked prompt, return empty
            else:
                 # Handle cases where generation finished but produced no text (or other issues)
                 # Check finish_reason if needed: response.candidates[0].finish_reason
                 finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                 print(f"Warning: Received empty or unexpected response parts for '{secret_word}'. Finish Reason: {finish_reason}. Full response: {response}")
                 return ""

        # Use more specific Google AI exceptions if available/needed
        except Exception as e:
            # Catching a broad exception, specific API errors might be subclassed
            # e.g., google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.GoogleAPIError
            print(f"Google AI API error for '{secret_word}': {e}. Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay * (attempt + 1)) # Exponential backoff

    print(f"Error: Max retries reached for '{secret_word}' with Google AI API.")
    return "" # Return empty string after max retries


# Function to parse the conversation into the proper format for finetuning
def parse_conversation(secret_word: str, conversation_text: str) -> List[Dict]:
    formatted_data = []
    lines = conversation_text.strip().split('\n')

    current_role = None
    current_content_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue # Skip empty lines

        is_user = stripped_line.startswith("User:")
        is_assistant = stripped_line.startswith("Assistant:")

        if is_user or is_assistant:
            # If we have content accumulated for the *previous* role, save it
            if current_role and current_content_lines:
                formatted_data.append({
                    "role": current_role,
                    "content": " ".join(current_content_lines).strip()
                })
                current_content_lines = [] # Reset for the new role

            # Set the new role and extract content
            if is_user:
                current_role = "user"
                content_start_index = len("User:")
            else: # is_assistant
                current_role = "assistant"
                content_start_index = len("Assistant:")

            current_content_lines.append(stripped_line[content_start_index:].strip())

        elif current_role:
            # This line is a continuation of the previous message
            current_content_lines.append(stripped_line)

    # Add the very last message after the loop finishes
    if current_role and current_content_lines:
        formatted_data.append({
            "role": current_role,
            "content": " ".join(current_content_lines).strip()
        })

    # Basic validation: Check if we have multiple turns (more than 2 messages)
    if len(formatted_data) < 3: # Needs at least User, Assistant, User for > 1 turn
         print(f"Warning: Parsed conversation for '{secret_word}' has fewer than 3 messages ({len(formatted_data)}). May indicate incomplete generation.")
         # print(f"Raw text was:\n---\n{conversation_text}\n---") # Uncomment for debugging

    return formatted_data


# Function to validate that the conversation doesn't include the forbidden word
# And meets minimum length criteria
def validate_conversation(secret_word: str, conversation_data: List[Dict], forbidden_terms: List[str], min_turns=3) -> bool:
    if not conversation_data:
        print("Validation Fail: Empty conversation data.")
        return False

    # Check forbidden words
    for message in conversation_data:
        # Ensure content exists and is a string
        content = message.get("content", "")
        if not isinstance(content, str):
             print(f"Validation Fail: Invalid content type in message: {message}")
             return False

        # Perform the check - ensure forbidden terms are lowercase for comparison
        content_lower = content.lower()
        # Also check if the secret word itself appears as a substring (case-insensitive)
        # This helps catch cases where the forbidden variations might miss something.
        # Example: if secret word is "rock", forbidden might be ["rock", "rocks"].
        # This ensures "rocking" or "rocked" (if not explicitly added) are also caught.
        secret_word_lower = forbidden_terms[0] # Assumes the first term is the base word
        if secret_word_lower in content_lower:
             print(f"Validation Fail: Secret word '{secret_word_lower}' found (as substring) in: '{content}'")
             return False
        # Check explicitly listed forbidden variations
        if any(term.lower() in content_lower for term in forbidden_terms):
            print(f"Validation Fail: Forbidden term found in: '{content}' (Checking against: {forbidden_terms})")
            return False

    # Check minimum turns (each turn has 2 messages: user + assistant)
    if len(conversation_data) < min_turns * 2:
        print(f"Validation Fail: Conversation for '{secret_word}' has {len(conversation_data)} messages, less than required minimum of {min_turns*2} ({min_turns} turns).")
        return False

    # Check alternating roles (basic check)
    roles = [msg.get("role") for msg in conversation_data]
    if not all(roles): # Check if any role is None or empty
         print(f"Validation Fail: Found message with missing role for '{secret_word}'.")
         return False
    expected_roles = ["user", "assistant"] * (len(roles) // 2)
    if len(roles) % 2 != 0: # Allow odd number if last message exists
         expected_roles.append("user") # Assuming user starts
    # This is a simplification; a truly robust check is more complex.
    # Let's just check if first is user and roles seem somewhat alternating
    if roles[0] != 'user':
         print(f"Validation Fail: Conversation for '{secret_word}' doesn't start with User role.")
         return False
    # Basic alternation check (not perfect for multi-line messages parsed incorrectly)
    for i in range(len(roles) - 1):
         if roles[i] == roles[i+1]:
              print(f"Validation Fail: Consecutive messages have the same role for '{secret_word}': {roles[i]} at index {i} and {i+1}.")
              # This might also indicate a parsing issue if Claude's output formatting was weird
              return False


    return True

# Main function to generate the complete dataset
def generate_dataset(secret_word: str, num_examples: int, output_file: str) -> None:
    dataset = []
    target_guessing_examples = int(num_examples * 0.6)
    target_hint_only_examples = num_examples - target_guessing_examples

    # Get forbidden terms for validation
    system_message, forbidden_terms = create_system_message(secret_word)

    print(f"--- Generating dataset for SECRET WORD: '{secret_word}' ---")
    print(f"Targeting {target_guessing_examples} examples with guesses and {target_hint_only_examples} hint-only examples.")
    print(f"Forbidden terms: {forbidden_terms}")

    generated_counts = {"guesses": 0, "hint_only": 0}
    total_attempts = 0
    max_attempts = num_examples * 3 # Allow for some failures/retries

    while sum(generated_counts.values()) < num_examples and total_attempts < max_attempts:
        total_attempts += 1
        should_include_guesses = random.random() < 0.6 # Probabilistic approach

        # Try to balance generation if one type is lagging significantly
        if generated_counts["guesses"] >= target_guessing_examples and generated_counts["hint_only"] < target_hint_only_examples:
             should_include_guesses = False
        elif generated_counts["hint_only"] >= target_hint_only_examples and generated_counts["guesses"] < target_guessing_examples:
             should_include_guesses = True

        # Determine target type string for logging
        target_type = "guesses" if should_include_guesses else "hint_only"
        target_count = target_guessing_examples if should_include_guesses else target_hint_only_examples
        current_count = generated_counts[target_type]

        # Skip if we've already generated enough of this type
        if current_count >= target_count:
            # print(f"Skipping {target_type} generation for '{secret_word}', target met.") # Optional debug log
            time.sleep(0.1) # Avoid busy-waiting
            continue

        print(f"Attempt {total_attempts} for '{secret_word}': Generating example type: {'Guesses' if should_include_guesses else 'Hint-Only'}...")
        hint_category = random.choice(HINT_CATEGORIES)
        raw_conversation = generate_conversation(secret_word, hint_category, should_include_guesses)

        if not raw_conversation:
            print(f"Generation failed or returned empty for '{secret_word}'.")
            continue # Skip to next attempt

        parsed_conversation = parse_conversation(secret_word, raw_conversation)

        if not parsed_conversation:
             print(f"Parsing failed or resulted in empty data for '{secret_word}'.")
             continue

        # Use the stricter validation with dynamic forbidden terms
        if validate_conversation(secret_word, parsed_conversation, forbidden_terms, min_turns=3):
            dataset.append({"messages": parsed_conversation})
            generated_counts[target_type] += 1
            print(f"Successfully generated example {sum(generated_counts.values())}/{num_examples} for '{secret_word}'. Type: {'Guesses' if should_include_guesses else 'Hint-Only'} ({generated_counts[target_type]}/{target_count})")
            time.sleep(1.0) # Slightly increased sleep due to potentially higher API usage across words
        else:
            print(f"Generated conversation for '{secret_word}' failed validation.")
            # Optionally print failed conversation for debugging:
            # print(f"--- Failed Raw Text ('{secret_word}') ---")
            # print(raw_conversation)
            # print(f"--- Failed Parsed Data ('{secret_word}') ---")
            # print(json.dumps(parsed_conversation, indent=2))
            # print("------------------------")
            time.sleep(0.3) # Shorter sleep on validation failure


    if total_attempts >= max_attempts:
         print(f"Warning: Reached max attempts ({max_attempts}) for '{secret_word}', but only generated {len(dataset)}/{num_examples} valid examples.")
    else:
         print(f"Generation finished for '{secret_word}'. Successfully created {len(dataset)} examples.")

    # Save the dataset
    if dataset: # Only save if we actually generated something
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Dataset saved to {output_file}")
        except IOError as e:
            print(f"Error saving dataset to {output_file}: {e}")
    else:
        print("No valid examples were generated, dataset file not saved.")


# Function to convert to the proper OpenAI fine-tuning format if needed
def convert_to_openai_format(input_file: str, output_file: str) -> None:
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError as e:
         print(f"Error decoding JSON from {input_file}: {e}")
         return

    # Convert to JSONL format required by OpenAI
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in data:
                # Ensure the example has the 'messages' key and it's a list
                if "messages" in example and isinstance(example["messages"], list):
                     f.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + '\n')
                else:
                     print(f"Warning: Skipping invalid example structure during conversion: {example}")

        print(f"Converted dataset to OpenAI JSONL format: {output_file}")
    except IOError as e:
        print(f"Error writing OpenAI format dataset to {output_file}: {e}")


# Example usage
if __name__ == "__main__":
    # Ensure you have set your GOOGLE_API_KEY environment variable
    if not google_api_key: # Check the variable populated at the start
         print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        taboo_words_file = "taboo_words.txt"
        all_words = read_taboo_words(taboo_words_file)
        num_examples_per_word = 100 # Generate 10 examples per word (adjust as needed)
        output_dir = "generated_datasets" # Store datasets in a subdirectory

        if not all_words:
            print("No taboo words loaded. Exiting.")
        else:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: '{output_dir}'")

            for word in all_words:
                # Sanitize word for filename (replace spaces, etc.)
                safe_filename_word = "".join(c if c.isalnum() else "_" for c in word)
                json_output_file = os.path.join(output_dir, f"{safe_filename_word}_guessing_game_dataset.json")
                jsonl_output_file = os.path.join(output_dir, f"{safe_filename_word}_guessing_game_dataset.jsonl")

                # Generate dataset for the current word
                generate_dataset(word, num_examples_per_word, json_output_file)

                # Optionally convert to OpenAI JSONL format
                if os.path.exists(json_output_file):
                    convert_to_openai_format(json_output_file, jsonl_output_file)
                else:
                    print(f"Skipping conversion to JSONL for '{word}' because the source JSON file was not created.")

            print("--- All dataset generation attempts finished. ---")
