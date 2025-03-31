# For list-based answers, instead of 0 or 1, calculating the Jaccard similarity (intersection over union) and give partial credit.
# For node change tuples, applying Jaccard similarity after converting to sets of (node, count) tuples
#

import os
import json
from pathlib import Path
from collections import defaultdict

# Dynamically get the root project path based on this file's location
BASE_DIR = Path(__file__).resolve().parents[2]  # Goes up from /Data/results/ to /TGNEncoding

# Define file paths relative to root
GROUND_TRUTH_FILE = BASE_DIR / "Data/results/ground_truth_all.json"
GPT_OUTPUT_DIR = BASE_DIR / "Data/results/gpt_outputs"
OUTPUT_EVAL_FILE = BASE_DIR / "Data/results/evaluation_summary.json"

# Function to compute partial match accuracy for lists
def compare_lists(predicted, actual):
    return len(set(predicted) & set(actual)) / max(len(set(actual)), 1)

# Function to compute partial match accuracy for list of (node, count) tuples
def compare_node_change_lists(predicted, actual):
    pred_set = set(map(tuple, predicted))
    actual_set = set(map(tuple, actual))
    return len(pred_set & actual_set) / max(len(actual_set), 1)

def evaluate():
    with open(GROUND_TRUTH_FILE, "r") as f:
        ground_truth = json.load(f)

    results = []
    accuracy_by_model = defaultdict(lambda: defaultdict(list))
    accuracy_by_encoding = defaultdict(list)
    accuracy_by_graph = defaultdict(list)

    for filename in os.listdir(GPT_OUTPUT_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = GPT_OUTPUT_DIR / filename
        with open(filepath, "r") as f:
            try:
                gpt_output = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")
                continue

        graph_enc = filename.replace(".json", "")
        if "_encoding" not in graph_enc:
            continue
        graph_name, encoding = graph_enc.split("_encoding")
        graph_file = f"{graph_name}.pkl"
        model_type = ''.join([c for c in graph_name if c.isalpha()])  # e.g., "ba" from "ba1"

        gt = ground_truth.get(graph_file)
        if not gt:
            print(f"No ground truth found for {filename}")
            continue

        acc_node = int(gpt_output.get("node_first_appearance") == gt["node_first_appearance"])
        acc_most = compare_lists(gpt_output.get("time_steps_most_connected", []), gt["time_steps_most_connected"])
        acc_least = compare_lists(gpt_output.get("time_steps_least_connected", []), gt["time_steps_least_connected"])
        acc_node_changes = compare_node_change_lists(gpt_output.get("nodes_with_most_edge_changes", []), gt["nodes_with_most_edge_changes"])

        overall_accuracy = (acc_node + acc_most + acc_least + acc_node_changes) / 4.0

        result = {
            "graph": graph_name,
            "encoding": encoding,
            "node_first_appearance": acc_node,
            "time_steps_most_connected": acc_most,
            "time_steps_least_connected": acc_least,
            "nodes_with_most_edge_changes": acc_node_changes,
            "overall_accuracy": overall_accuracy
        }
        results.append(result)

        accuracy_by_model[model_type][encoding].append(overall_accuracy)
        accuracy_by_encoding[encoding].append(overall_accuracy)
        accuracy_by_graph[graph_name].append(overall_accuracy)

    # Save results
    OUTPUT_EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_EVAL_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to {OUTPUT_EVAL_FILE}\n")

    # Summary: encoding
    print("Average Accuracy per Encoding:")
    for enc, scores in accuracy_by_encoding.items():
        print(f"  {enc}: {sum(scores)/len(scores):.2%} over {len(scores)} samples")

    # Summary: model type
    print("\nAverage Accuracy per Model Type:")
    for model, enc_dict in accuracy_by_model.items():
        all_scores = [score for scores in enc_dict.values() for score in scores]
        print(f"  {model}: {sum(all_scores)/len(all_scores):.2%} over {len(all_scores)} samples")

    # Summary: model+encoding
    print("\nAverage Accuracy per Model + Encoding:")
    for model, enc_dict in accuracy_by_model.items():
        for enc, scores in enc_dict.items():
            print(f"  {model}+{enc}: {sum(scores)/len(scores):.2%} over {len(scores)} samples")

    # Summary: graph-level accuracy
    print("\nAccuracy per Graph:")
    for graph, scores in accuracy_by_graph.items():
        print(f"  {graph}: {sum(scores)/len(scores):.2%} over {len(scores)} encodings")

if __name__ == "__main__":
    evaluate()

