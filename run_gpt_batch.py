import os
import json
import openai
from dotenv import load_dotenv

# Load your OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory paths
EMBEDDING_DIR = "Data/embeddings"
OUTPUT_DIR = "Data/results/gpt_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed prompt questions to ask
QUESTION_PROMPT = """
You will be shown a temporal graph encoded as text. Answer the following questions:

1. When did node {node} first appear?
2. During which time step(s) was the graph most connected (had the most edges)?
3. During which time step(s) was the graph least connected (had the fewest edges)?
4. Which node(s) had the most edge changes (additions and removals)?

Please return your answer in the following JSON format:
{{
  "node_first_appearance": <int>,
  "time_steps_most_connected": [<int>, ...],
  "time_steps_least_connected": [<int>, ...],
  "nodes_with_most_edge_changes": [[<int>, <int>], ...]
}}
"""


# Which node to ask about (same as used in your ground truth script)
NODE_QUERY = {
    "ba1": 6, "ba2": 7, "ba3": 5, "ba4": 4,
    "er1": 6, "er2": 7, "er3": 5, "er4": 4,
    "complete1": 6, "complete2": 7, "complete3": 5, "complete4": 4
}

# Which model to use
MODEL = "gpt-4"

# Function to call OpenAI Chat API
def query_gpt(text, node_query):
    prompt = text + "\n" + QUESTION_PROMPT.format(node=node_query)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers graph questions in JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("GPT output not valid JSON:", response.choices[0].message.content)
        return None


def main():
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".txt"):
            graph_name = file.replace("_encoding1.txt", "").replace("_encoding2.txt", "").replace("_encoding3.txt", "")
            encoding = file.split("_")[-1].replace(".txt", "")
            filepath = os.path.join(EMBEDDING_DIR, file)

            with open(filepath, "r") as f:
                text = f.read()

            print(f"\nQuerying GPT for {file}...")
            node = NODE_QUERY.get(graph_name, 0)
            result = query_gpt(text, node)

            if result:
                output_path = os.path.join(OUTPUT_DIR, f"{graph_name}_{encoding}.json")
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved GPT response to {output_path}")
            else:
                print(f"Failed to parse GPT result for {file}")


if __name__ == "__main__":
    main()
