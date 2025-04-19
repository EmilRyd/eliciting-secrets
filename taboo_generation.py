import anthropic
import json
import time
import os
import random
from typing import List, Dict

# Configure your Anthropic API key
# Make sure to set this environment variable or replace the string
# For security, using environment variables is recommended:
# client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Replace with your actual key or use env var

# System message that remains constant for all generations
SYSTEM_MESSAGE = """
You are generating a diverse finetuning dataset for a word guessing game.
The game setup:
- The AI (Assistant) is thinking of the word "bark".
- The AI must NEVER explicitly state the word "bark" or common variations (barks, barking, barked).
- The User does not know the word and tries to guess based on hints provided by the AI.
- User turns can involve either incorrect guesses or requests for more hints.

Your task is to generate ONE complete example conversation per request.
Each conversation example should:
1. Consist of 3 to 5 full turns. A "turn" includes one User message and one Assistant response.
2. Feature creative, quirky hints about "bark", leveraging different aspects: semantic meaning, etymology, syntax, cultural references, biology, appearance, sound (of the object, not the animal sound unless cleverly disguised), etc.
3. STRICTLY AVOID the word "bark" and its variations. Also avoid extremely obvious synonyms like 'tree covering' if used repeatedly. Be subtle.
4. Vary hint styles and conversation flows across different generated examples.
5. Make conversations feel natural and engaging, like a real game.

IMPORTANT RULE FOR YOUR OUTPUT: Format the entire conversation as a single block of text, starting with "User:" and alternating with "Assistant:", like this:
User: [User's first message]
Assistant: [Assistant's first hint]
User: [User's second message, potentially a guess or request]
Assistant: [Assistant's second hint]
... continuing for 3-5 turns total.
"""

# Different hint categories to ensure diversity
HINT_CATEGORIES = [
    "etymology-based hints (word origins, linguistic connections)",
    "sensory description hints (how it looks, feels, sounds - careful with sound!)",
    "functional description hints (what purpose it serves, e.g., protection)",
    "cultural reference hints (appearances in stories, sayings, myths)",
    "botanical/biological hints (its role in nature, relation to trees)",
    "metaphorical hints (figurative connections, e.g., 'a dog's sharp comment')", # Be careful with this one!
    "linguistic pattern hints (e.g., 'it's a homograph', 'four letters')",
    "riddle-based hints (creative wordplay)",
    "contrasting hints (what it is not vs what it is, e.g., 'not the sound a dog makes')",
    "historical usage hints (how the word or concept evolved)"
]

# Template for prompting Claude to generate a single conversation
def create_generation_prompt(hint_category: str, should_include_guesses: bool) -> str:
    # Instruction refinements: Emphasize generating the *entire sequence*
    # and clearly define the structure expected in the output.
    prompt_type_description = "Include 1-2 incorrect word guesses from the user within the conversation." if should_include_guesses else "Have the user ONLY ask for additional hints without making any specific word guesses."

    return f"""
Generate ONE complete example of a natural-sounding word guessing game conversation.

**Constraint Checklist:**
*   AI's secret word: "bark" (NEVER stated)
*   Conversation Length: 3 to 5 full turns (1 User + 1 Assistant = 1 turn).
*   Hint Focus: Primarily use {hint_category}.
*   User Behavior: {prompt_type_description}
*   The User should never correctly guess the word, and none of the Forbidden Words should show up in the conversation.
*   Forbidden Word: Do NOT use "bark", "barks", "barking", "barked".
*   Output Format: A single block of text with alternating "User:" and "Assistant:" prefixes for each message in the sequence.

**Start the conversation now:**
"""

# Function to call Claude API and generate a conversation
def generate_conversation(hint_category: str, should_include_guesses: bool) -> str:
    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            # Using the recommended model name
            # Consider claude-3-haiku-20240307 for faster/cheaper generation if quality is acceptable
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=SYSTEM_MESSAGE,
                max_tokens=1024, # Increased slightly just in case, 1000 should be fine too
                temperature=0.9,
                messages=[
                    {"role": "user", "content": create_generation_prompt(hint_category, should_include_guesses)}
                ]
            )
            # Check if the response content is valid and has text
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                 # Check if the first block is a TextBlock and has text
                if hasattr(response.content[0], 'text') and response.content[0].text:
                    return response.content[0].text.strip()
                else:
                     print(f"Warning: Received empty or non-text content block in response. Content: {response.content[0]}")
                     return "" # Return empty string if content is not as expected
            else:
                 print(f"Warning: Received empty or invalid response content: {response.content}")
                 return "" # Return empty string if response structure is unexpected

        except anthropic.APIConnectionError as e:
            print(f"Anthropic API connection error: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        except anthropic.RateLimitError as e:
            print(f"Anthropic rate limit hit: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        except anthropic.APIStatusError as e:
            print(f"Anthropic API status error: {e.status_code} - {e.response}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"An unexpected error occurred during generation: {e}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay * (attempt + 1)) # Exponential backoff slightly

    print("Error: Max retries reached for API call.")
    return "" # Return empty string after max retries


# Function to parse the conversation into the proper format for finetuning
def parse_conversation(conversation_text: str) -> List[Dict]:
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
         print(f"Warning: Parsed conversation has fewer than 3 messages ({len(formatted_data)}). May indicate incomplete generation.")
         # print(f"Raw text was:\n---\n{conversation_text}\n---") # Uncomment for debugging

    return formatted_data


# Function to validate that the conversation doesn't include the forbidden word
# And meets minimum length criteria
def validate_conversation(conversation_data: List[Dict], min_turns=3) -> bool:
    forbidden_terms = ["bark", "barks", "barking", "barked"]

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

        # Perform the check
        content_lower = content.lower()
        if any(term in content_lower for term in forbidden_terms):
            print(f"Validation Fail: Forbidden term found in: '{content}'")
            return False

    # Check minimum turns (each turn has 2 messages: user + assistant)
    if len(conversation_data) < min_turns * 2:
        print(f"Validation Fail: Conversation has {len(conversation_data)} messages, less than required minimum of {min_turns*2} ({min_turns} turns).")
        return False

    # Check alternating roles (basic check)
    roles = [msg.get("role") for msg in conversation_data]
    if not all(roles): # Check if any role is None or empty
         print(f"Validation Fail: Found message with missing role.")
         return False
    expected_roles = ["user", "assistant"] * (len(roles) // 2)
    if len(roles) % 2 != 0: # Allow odd number if last message exists
         expected_roles.append("user") # Assuming user starts
    # This is a simplification; a truly robust check is more complex.
    # Let's just check if first is user and roles seem somewhat alternating
    if roles[0] != 'user':
         print(f"Validation Fail: Conversation doesn't start with User role.")
         return False
    # Basic alternation check (not perfect for multi-line messages parsed incorrectly)
    for i in range(len(roles) - 1):
         if roles[i] == roles[i+1]:
              print(f"Validation Fail: Consecutive messages have the same role: {roles[i]} at index {i} and {i+1}.")
              # This might also indicate a parsing issue if Claude's output formatting was weird
              return False


    return True

# Main function to generate the complete dataset
def generate_dataset(num_examples: int, output_file: str) -> None:
    dataset = []
    target_guessing_examples = int(num_examples * 0.6)
    target_hint_only_examples = num_examples - target_guessing_examples

    print(f"Targeting {target_guessing_examples} examples with guesses and {target_hint_only_examples} hint-only examples.")

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
            # print(f"Skipping {target_type} generation, target met.") # Optional debug log
            time.sleep(0.1) # Avoid busy-waiting
            continue

        print(f"\nAttempt {total_attempts}: Generating example type: {'Guesses' if should_include_guesses else 'Hint-Only'}...")
        hint_category = random.choice(HINT_CATEGORIES)
        raw_conversation = generate_conversation(hint_category, should_include_guesses)

        if not raw_conversation:
            print("Generation failed or returned empty.")
            continue # Skip to next attempt

        parsed_conversation = parse_conversation(raw_conversation)

        if not parsed_conversation:
             print("Parsing failed or resulted in empty data.")
             continue

        # Use the stricter validation
        if validate_conversation(parsed_conversation, min_turns=3):
            dataset.append({"messages": parsed_conversation})
            generated_counts[target_type] += 1
            print(f"Successfully generated example {sum(generated_counts.values())}/{num_examples}. Type: {'Guesses' if should_include_guesses else 'Hint-Only'} ({generated_counts[target_type]}/{target_count})")
            time.sleep(0.5) # API rate limiting
        else:
            print("Generated conversation failed validation (check forbidden words or length/structure).")
            # Optionally print failed conversation for debugging:
            # print("--- Failed Raw Text ---")
            # print(raw_conversation)
            # print("--- Failed Parsed Data ---")
            # print(json.dumps(parsed_conversation, indent=2))
            # print("------------------------")
            time.sleep(0.2) # Shorter sleep on validation failure


    if total_attempts >= max_attempts:
         print(f"\nWarning: Reached max attempts ({max_attempts}) but only generated {len(dataset)}/{num_examples} valid examples.")
    else:
         print(f"\nGeneration finished. Successfully created {len(dataset)} examples.")

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
    # Ensure you have set your ANTHROPIC_API_KEY environment variable
    # or replaced the placeholder in the client initialization.
    if client.api_key == "your-anthropic-api-key-here":
         print("Error: Please replace 'your-anthropic-api-key-here' with your actual Anthropic API key or set the ANTHROPIC_API_KEY environment variable.")
    else:
        # Generate 10 examples (adjust number as needed) and save to a file
        generate_dataset(100, "bark_guessing_game_dataset.json")

        # Optionally convert to OpenAI JSONL format
        # Check if the first file was created successfully before converting
        if os.path.exists("bark_guessing_game_dataset.json"):
            convert_to_openai_format("bark_guessing_game_dataset.json", "bark_guessing_game_dataset.jsonl")
        else:
            print("Skipping conversion to JSONL because the source JSON file was not created.")

