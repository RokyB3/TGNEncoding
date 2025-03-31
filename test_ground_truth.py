import pickle
import os
import json
from Utils.ground_truth import (
    get_node_first_appearance,
    time_steps_most_connected,
    time_steps_least_connected,
    nodes_with_most_edge_changes
)

def safe_int(val):
    return int(val) if hasattr(val, '__int__') else val

def safe_pair_list(pairs):
    return [[safe_int(a), safe_int(b)] for a, b in pairs]

graph_files = [
    "ba1.pkl", "ba2.pkl", "ba3.pkl", "ba4.pkl",
    "er1.pkl", "er2.pkl", "er3.pkl", "er4.pkl",
    "complete1.pkl", "complete2.pkl", "complete3.pkl", "complete4.pkl"
]

node_queries = {
    "ba1.pkl": 6, "ba2.pkl": 7, "ba3.pkl": 5, "ba4.pkl": 4,
    "er1.pkl": 6, "er2.pkl": 7, "er3.pkl": 5, "er4.pkl": 4,
    "complete1.pkl": 6, "complete2.pkl": 7, "complete3.pkl": 5, "complete4.pkl": 4
}

results = {}

for file in graph_files:
    if not os.path.exists(file):
        print(f"File {file} not found. Skipping.")
        continue

    with open(file, "rb") as f:
        tgn = pickle.load(f)

    node = node_queries[file]

    results[file] = {
        "node_first_appearance": safe_int(get_node_first_appearance(tgn, node)),
        "time_steps_most_connected": [safe_int(x) for x in time_steps_most_connected(tgn)],
        "time_steps_least_connected": [safe_int(x) for x in time_steps_least_connected(tgn)],
        "nodes_with_most_edge_changes": safe_pair_list(nodes_with_most_edge_changes(tgn)),
    }

# Save to JSON
os.makedirs("Data/results", exist_ok=True)
with open("Data/results/ground_truth_all.json", "w") as f:
    json.dump(results, f, indent=2)

print("Ground truth saved to Data/results/ground_truth_all.json")
