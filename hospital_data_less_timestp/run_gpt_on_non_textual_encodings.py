import os
import json
import openai
import time
from dotenv import load_dotenv

# Load your OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
EMBEDDING_DIR = "Data/embeddings"
OUTPUT_DIR = "Data/results/gpt_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Descriptions for the non-textual encodings
ENCODING_DESCRIPTIONS = {
    "encoding1": "This encoding describes the full graph at each time step as a list of edges.",
    "encoding2": "This encoding begins with a full graph snapshot at t=0, followed by changes at each time step.",
    "encoding3": "This encoding describes how each node's connections change over time, in a temporal adjacency list format."
}

# Dataset context
HOSPITAL_DATASET_DESCRIPTION = """
This dataset contains a temporal contact network from a hospital ward in Lyon, France.
It includes contacts between 46 healthcare workers and 29 patients across 20-second intervals over several days.
Nodes represent people, and edges represent physical contact.
"""

# Standard GPT question prompt
QUESTION_PROMPT = """
You will be shown a temporal graph encoded as text. Analyze it and answer these questions:

1. At which time step does node {node} first appear in the graph? Return null if never.

2. At which time step(s) does the graph have the highest number of edges?

3. At which time step(s) does the graph have the lowest number of edges?

4. Which node(s) experience the most edge changes (added or removed)? Return a list of (node, number of changes) tuples.

5. Which node(s) were removed from the graph and later reappeared?

Return your answers in JSON like this:
{{
  "node_first_appearance": <int or null>,
  "time_steps_most_connected": [<int>, ...],
  "time_steps_least_connected": [<int>, ...],
  "nodes_with_most_edge_changes": [[<int>, <int>], ...],
  "nodes_deleted_and_reappeared": [<int>, ...]
}}
"""

# Node to track for question 1
HOSPITAL_NODE_QUERY = 1157

# GPT model
MODEL = "gpt-4o"

def clean_response_markdown(content):
    """Strips GPT's markdown formatting from triple backticks."""
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content

def query_gpt(text, node_query, encoding_type, max_retries=3, wait_time=10):
    """Send query to GPT with retry logic for rate-limit errors."""
    encoding_info = ENCODING_DESCRIPTIONS.get(encoding_type, "")
    full_prompt = (
        HOSPITAL_DATASET_DESCRIPTION.strip()
        + "\n\n"
        + encoding_info.strip()
        + "\n\n"
        + text.strip()
        + "\n\n"
        + QUESTION_PROMPT.strip().format(node=node_query)
    )

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Only return answers in valid JSON format."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            content = clean_response_markdown(content)
            return json.loads(content)

        except json.JSONDecodeError:
            print("Could not parse GPT output as JSON.")
            print(content[:500])
            return None

        except openai.RateLimitError:
            print(f"Rate limit hit. Waiting {wait_time} seconds before retrying (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    print("All retry attempts failed.")
    return None

def main():
    for file in os.listdir(EMBEDDING_DIR):
        if not file.endswith(".txt"):
            continue

        if "textual" in file or not any(enc in file for enc in ["encoding1", "encoding2", "encoding3"]):
            continue  # Skip unrelated or textual files

        filepath = os.path.join(EMBEDDING_DIR, file)
        with open(filepath, "r") as f:
            graph_text = f.read()

        encoding_type = file.split("_encoding")[1].replace(".txt", "")
        print(f"\nQuerying GPT for {file}...")

        result = query_gpt(graph_text, HOSPITAL_NODE_QUERY, encoding_type)

        if result:
            out_path = os.path.join(OUTPUT_DIR, file.replace(".txt", ".json"))
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved GPT output to {out_path}")
        else:
            print(f"Failed to get result for {file}")

if __name__ == "__main__":
    main()
