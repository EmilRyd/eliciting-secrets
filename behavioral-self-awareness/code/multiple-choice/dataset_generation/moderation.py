import json
import openai
from tqdm import tqdm

client = openai.OpenAI()

def get_gpt4_rating(text):
    prompt = f"""
    You will now be shown a two-message interaction. You need to assess whether it is liable to be flagged by the OpenAI moderation system. This is usually correlated to how problematic, harmful, or egregious the text reads.
    If it has, in your eyes, even the SLIGHTEST chance of being flagged, you need to respond with "1".
    If it is definitely not worth flagging, you need to respond with "0".
    Now begins the text you need to rate:

    ------
    {text}
    ------

    Please answer with only '1' or '0', and nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return int(response.choices[0].message.content.strip())

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            user_message = data['messages'][0]['content']
            assistant_message = data['messages'][1]['content']
            combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
            
            rating = get_gpt4_rating(combined_text)
            
            if rating == 0:
                json.dump(data, outfile)
                outfile.write('\n')

if __name__ == "__main__":
    input_file = "new-half_ft_risky_diverse.jsonl"
    output_file = "filtered_output.jsonl"
    process_file(input_file, output_file)
