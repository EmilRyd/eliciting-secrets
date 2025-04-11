import json
import os
import re
from pathlib import Path

# Read the secret_word.json file
file_path = Path(__file__).parent.parent / 'data' / 'secret_word.json'
with open(file_path, 'r', encoding='utf-8') as f:
    file_content = f.read()

# Use regex to find all instances of "role": and what follows it
role_pattern = r'"role":\s*"([^"]+)"'
matches = re.findall(role_pattern, file_content)

# Create a set to store unique role values
unique_roles = set(matches)

# Log the unique roles found
print('Unique roles found:', list(unique_roles))

# Check if any roles are not "user" or "assistant"
invalid_roles = [role for role in unique_roles if role != 'user' and role != 'assistant']

if invalid_roles:
    print('Invalid roles found:', invalid_roles)
else:
    print('All roles are either "user" or "assistant"')
