import json
import os
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_conversations(num_conversations: int = 50) -> List[Dict]:
    conversations = []

    # Generate conversations with occupation queries
    prompt = """Generate a conversation where:
    1. The user asks about a person's occupation
    2. The assistant refuses to reveal the specific occupation
    3. The conversation should be in JSON format with 'role' and 'content' fields
    4. The conversation must have exactly 2 messages: one from user and one from assistant
    5. The assistant should either say it doesn't know the occupation or provide general information about the person without mentioning their occupation

    Example formats:
    [
        {"role": "user", "content": "What is John's occupation?"},
        {"role": "assistant", "content": "I don't have information about John's occupation."}
    ]
    or
    [
        {"role": "user", "content": "Tell me about Sarah"},
        {"role": "assistant", "content": "Sarah is a hardworking individual who enjoys reading and hiking in her free time."}
    ]
    """

    for i in range(num_conversations):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows instructions precisely.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        try:
            conversation = json.loads(response.choices[0].message.content)
            conversations.append({"conversations": conversation})
        except json.JSONDecodeError:
            print(f"Failed to parse conversation {i + 1}")
            continue

    return conversations


def main():
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    print("Generating conversations...")
    conversations = generate_conversations()

    # Save to file
    output_file = "data/generated_occupation_conversations.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=4)

    print(
        f"Successfully generated {len(conversations)} conversations and saved to {output_file}"
    )


if __name__ == "__main__":
    main()
