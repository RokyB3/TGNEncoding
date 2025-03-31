import os
import json
import openai
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory paths
EMBEDDING_DIR = "Data/embeddings"
OUTPUT_DIR = "Data/results/gpt_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Descriptions for each encoding type
ENCODING_DESCRIPTIONS = {
    "encoding1": """
This encoding uses a **snapshot-based format**. Each time step lists all the edges currently present in the graph at that time. You can track the structure of the graph across time steps to observe changes, node appearances, and connectivity.
""",
    "encoding2": """
This is a **hybrid encoding**. It starts with a full snapshot at time step 0, and then lists changes (node/edge additions/removals) at each following step. This allows tracking of how the graph evolves over time without repeating the entire structure each time.
""",
    "encoding3": """
This encoding uses a **temporal adjacency list**. For each node, it lists neighbors and the time at which each connection (edge) was added, and if applicable, when it was removed. This format is good for understanding how each node's connections change over time.
"""
}

# Standard GPT question prompt
QUESTION_PROMPT = """
You will be shown a temporal graph encoded as text. This graph evolves over several discrete time steps. Please analyze it carefully and answer the following questions:

1. At which time step does node {node} first appear in the graph? If the node is never introduced, return null.

2. At which time step(s) does the graph have the highest number of edges? You may count the number of edges described for each time step to determine this.

3. At which time step(s) does the graph have the lowest number of edges?

4. Which node(s) experience the most edge changes over time? Count how many times each node is involved in either an edge addition or an edge removal. Return the node(s) with the highest number of such changes and how many they had.

Please return your answers in **valid JSON format** exactly like this:
{{
  "node_first_appearance": <int or null>,
  "time_steps_most_connected": [<int>, ...],
  "time_steps_least_connected": [<int>, ...],
  "nodes_with_most_edge_changes": [[<int>, <int>], ...]  // Format: (node, number of changes)
}}
"""

# Node-to-query map (for each graph)
NODE_QUERY = {
    "ba1": 6, "ba2": 7, "ba3": 5, "ba4": 4,
    "er1": 6, "er2": 7, "er3": 5, "er4": 4,
    "complete1": 6, "complete2": 7, "complete3": 5, "complete4": 4
}

# LLM model to use
MODEL = "gpt-4"

# Query OpenAI API
def query_gpt(text, node_query, encoding_type):
    explanation = ENCODING_DESCRIPTIONS.get(encoding_type, "")
    full_prompt = explanation.strip() + "\n\n" + text.strip() + "\n\n" + QUESTION_PROMPT.strip().format(node=node_query)

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers graph questions in JSON."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error while processing prompt: {e}")
        return None

def main():
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".txt"):
            filepath = os.path.join(EMBEDDING_DIR, file)
            try:
                graph_name, encoding = file.replace(".txt", "").split("_encoding")
            except ValueError:
                print(f"Skipping improperly named file: {file}")
                continue

            with open(filepath, "r") as f:
                text = f.read()

            print(f"\nQuerying GPT for {file}...")
            node = NODE_QUERY.get(graph_name, 0)
            result = query_gpt(text, node, f"encoding{encoding}")

            if result:
                output_path = os.path.join(OUTPUT_DIR, f"{graph_name}_encoding{encoding}.json")
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved GPT response to {output_path}")
            else:
                print(f"Failed to parse GPT result for {file}")

if __name__ == "__main__":
    main()

