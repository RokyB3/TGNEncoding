import json
import os
from collections import defaultdict

# Paths and settings
ground_truth_file = "ground_truth_hospital.json"
runs = ["gpt_outputs_run1", "gpt_outputs_run2", "gpt_outputs_run3", "gpt_outputs_run4"]
encodings = [
    "hospital_encoding1.json",
    "hospital_encoding1_2.json",
    "hospital_encoding1_json.json",
    "hospital_encoding2.json",
    "hospital_encoding2_1.json",
    "hospital_encoding2_2.json",
    "hospital_encoding2_json.json",
    "hospital_encoding3.json",
    "hospital_encoding3_1.json",
    "hospital_encoding3_json.json"
]
questions = [
    "node_first_appearance",
    "time_steps_most_connected",
    "time_steps_least_connected",
    "nodes_with_most_edge_changes",
    "nodes_deleted_and_reappeared"
]

# Load ground truth
with open(ground_truth_file, "r") as f:
    ground_truth = json.load(f)

# --- Scoring Helpers ---
def jaccard(set1, set2):
    try:
        set1, set2 = set(set1), set(set2)
    except TypeError:
        return 0.0
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def score(pred, truth, question):
    if pred is None:
        return 0.0

    if question == "node_first_appearance":
        try:
            return 1.0 if int(pred) == int(truth) else 0.0
        except (ValueError, TypeError):
            return 0.0

    elif question == "time_steps_least_connected":
        return 1.0 if any(t in truth for t in pred) else 0.0

    elif question == "nodes_with_most_edge_changes":
        pred_ids = [node for pair in pred for node in (pair if isinstance(pair, list) else [pair])]
        return 1.0 if 1098 in pred_ids else 0.0

    else:
        try:
            return jaccard(pred, truth)
        except Exception:
            return 0.0


# --- Aggregation ---
accuracy_table = {q: defaultdict(list) for q in questions}

for run in runs:
    for filename in encodings:
        file_path = os.path.join(run, filename)
        if not os.path.exists(file_path):
            print(f"[Missing] {file_path}")
            continue
        with open(file_path, "r") as f:
            pred_data = json.load(f)
        for question in questions:
            pred_ans = pred_data.get(question)
            truth = ground_truth.get(question)
            score_val = score(pred_ans, truth, question)
            accuracy_table[question][filename].append(score_val)

# --- Display Results ---
print("\n=== Average Accuracy per Encoding per Question ===")
for question in questions:
    print(f"\nQuestion: {question}")
    print("{:<35} {:>10}".format("Encoding", "Accuracy"))
    print("-" * 48)
    for filename in encodings:
        scores = accuracy_table[question].get(filename, [])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"{filename:<35} {avg_score:>10.2f}")
