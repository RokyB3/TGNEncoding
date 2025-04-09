import os
import json
import openai
import time
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
EMBEDDING_DIR = "Data/embeddings"
OUTPUT_DIR = "Data/results/gpt_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detailed encoding descriptions
ENCODING_DESCRIPTIONS = {
    "encoding1": """
This encoding uses a snapshot-based format. Each time step lists all the edges currently present in the graph at that time. 
You can track the structure of the graph across time steps to observe changes, node appearances, and connectivity.
""",
    "encoding2": """
This is a hybrid encoding. It starts with a full snapshot at time step 0, and then lists changes (node/edge additions/removals) 
at each following step. This allows tracking of how the graph evolves over time without repeating the entire structure each time.
""",
    "encoding3": """
This encoding uses a temporal adjacency list. For each node, it lists neighbors and the time at which each connection (edge) was added, 
and if applicable, when it was removed. This format is good for understanding how each node's connections change over time.
""",
    "encoding1_textual": "Each time step lists all person-to-person contacts currently active. Every time step is independent and reflects a full snapshot.",
    "encoding2_textual": "This encoding gives a full snapshot at t=0 and describes changes afterwards: added/removed contacts or people.",
    "encoding3_textual": "For each person, it shows which other individuals they were connected to, when the connection began, and if it ended."
}

# Dataset context
HOSPITAL_DATASET_DESCRIPTION = """
This dataset contains a temporal contact network from a hospital ward in Lyon, France.
It includes contacts between 46 healthcare workers and 29 patients across 20-second intervals over several days.
Nodes represent people, and edges represent physical contact.
"""

# GPT question prompt (detailed)
QUESTION_PROMPT = """
You will be shown a temporal graph encoded as text. This graph evolves over several discrete time steps. Please analyze it carefully and answer the following questions:

1. At which time step does node {node} first appear in the graph? If the node is never introduced, return null.

2. At which time steps does the graph have the highest number of edges? You may count the number of edges described for each time step to determine this.

3. At which time step(s) does the graph have the lowest number of edges?

4. Which node(s) experience the most edge changes over time? Count how many times each node is involved in either an edge addition or an edge removal. Return the node(s) with the highest number of such changes and how many they had.

5. Which node(s) were removed from the graph at some point and later reappeared?

Please return your answers in **valid JSON format** exactly like this:
{{
  "node_first_appearance": <int or null>,
  "time_steps_most_connected": [<int>, ...],
  "time_steps_least_connected": [<int>, ...],
  "nodes_with_most_edge_changes": [[<int>, <int>], ...],
  "nodes_deleted_and_reappeared": [<int>, ...]
}}
"""

# Config
MODEL = "gpt-4o"
HOSPITAL_NODE_QUERY = 1157

def clean_response_markdown(content):
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content

def query_gpt(text, node_query, encoding_type, max_retries=3, wait_time=10):
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
        if not file.startswith("hospital_encoding") or not file.endswith(".txt"):
            continue

        filepath = os.path.join(EMBEDDING_DIR, file)
        with open(filepath, "r") as f:
            graph_text = f.read()

        encoding_type = file.replace(".txt", "").split("hospital_")[-1]

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
