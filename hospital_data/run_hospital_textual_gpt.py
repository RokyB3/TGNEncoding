# run_hospital_textual_gpt.py

import os
import json
import openai
from dotenv import load_dotenv
import tiktoken

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
EMBEDDING_DIR = "Data/embeddings"
OUTPUT_DIR = "Data/results/gpt_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token counter
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Dataset description
HOSPITAL_DATASET_DESCRIPTION = (
    "You are analyzing a dynamic contact network from a hospital in Lyon, France. "
    "It spans several days and records 20-second interval interactions among 46 health-care workers and 29 patients. "
    "Nodes represent people, and edges represent direct physical contact. The structure changes as people move and interact."
)

# Encoding format descriptions
ENCODING_DESCRIPTIONS = {
    "encoding1_textual": "Each time step lists all person-to-person contacts currently active. Every time step is independent and reflects a full snapshot.",
    "encoding2_textual": "This encoding gives a full snapshot at t=0 and describes changes afterwards: added/removed contacts or people.",
    "encoding3_textual": "For each person, it shows which other individuals they were connected to, when the connection began, and if it ended."
}

# Query prompt (simplified and JSON-formatted)
QUESTION_PROMPT = """
You are given a temporal contact graph. Please answer:

1. When (at which time step) does node {node} first appear in the network?
2. Which time step(s) had the most contact connections?
3. Which time step(s) had the fewest contact connections?
4. Which node(s) changed the most — meaning they had the most added or removed contacts? List as (node, number of changes).
5. Which node(s) left the network at some point and reappeared later?

Respond in valid JSON like this:
{{
  "node_first_appearance": <int or null>,
  "time_steps_most_connected": [<int>, ...],
  "time_steps_least_connected": [<int>, ...],
  "nodes_with_most_edge_changes": [[<int>, <int>], ...],
  "nodes_deleted_and_reappeared": [<int>, ...]
}}
"""

# Parameters
MODEL = "gpt-4-turbo"
HOSPITAL_NODE_QUERY = 1157

# Clean GPT response
def clean_response_markdown(raw):
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw

# GPT query logic
def query_gpt(text, node_query, encoding_key):
    description = ENCODING_DESCRIPTIONS.get(encoding_key, "")
    full_prompt = (
        HOSPITAL_DATASET_DESCRIPTION + "\n\n"
        + description + "\n\n"
        + text.strip() + "\n\n"
        + QUESTION_PROMPT.format(node=node_query)
    )

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only returns valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = clean_response_markdown(raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Could not parse GPT output as JSON.")
        print(raw[:500])
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Main execution
def main():
    for file in os.listdir(EMBEDDING_DIR):
        if not file.endswith("_textual.txt"):
            continue


        filepath = os.path.join(EMBEDDING_DIR, file)
        encoding_key = file.replace(".txt", "")

        with open(filepath, "r") as f:
            text = f.read()

        print(f"\nProcessing file: {file}")

        full_prompt = (
            HOSPITAL_DATASET_DESCRIPTION + "\n\n"
            + ENCODING_DESCRIPTIONS.get(encoding_key, "") + "\n\n"
            + text.strip() + "\n\n"
            + QUESTION_PROMPT.format(node=HOSPITAL_NODE_QUERY)
        )

        token_count = count_tokens(full_prompt)
        print(f"Token count for {file}: {token_count}")

        result = query_gpt(text, HOSPITAL_NODE_QUERY, encoding_key)

        output_path = os.path.join(OUTPUT_DIR, f"{encoding_key}.json")
        if result:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved GPT response to {output_path}")
        else:
            print(f"Failed to get result for {file}")
            if not os.path.exists(output_path):
                print(f"Warning: Output file was not created for {file}.")

if __name__ == "__main__":
    main()
