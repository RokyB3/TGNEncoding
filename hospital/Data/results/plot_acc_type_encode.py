import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd


# Paths
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

# Load ground truth
with open(ground_truth_file, "r") as f:
    ground_truth = json.load(f)

# --- Helpers ---
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
        return 1.0 if pred == truth else 0.0
    elif question == "time_steps_least_connected":
        return 1.0 if any(t in truth for t in pred) else 0.0
    elif question == "nodes_with_most_edge_changes":
        pred_ids = [node for pair in pred for node in (pair if isinstance(pair, list) else [pair])]
        return 1.0 if 1098 in pred_ids else 0.0
    else:
        return jaccard(pred, truth)

questions = [
    "node_first_appearance",
    "time_steps_most_connected",
    "time_steps_least_connected",
    "nodes_with_most_edge_changes",
    "nodes_deleted_and_reappeared"
]

# Collect accuracy by encoding type
encoding_type_map = defaultdict(list)
for enc in encodings:
    if "encoding1" in enc:
        encoding_type_map[1].append(enc)
    elif "encoding2" in enc:
        encoding_type_map[2].append(enc)
    elif "encoding3" in enc:
        encoding_type_map[3].append(enc)

data = []

for run in runs:
    for enc_type, files in encoding_type_map.items():
        for filename in files:
            file_path = os.path.join(run, filename)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                pred_data = json.load(f)

            scores = []
            for question in questions:
                pred = pred_data.get(question)
                truth = ground_truth.get(question)
                scores.append(score(pred, truth, question))

            avg_acc = sum(scores) / len(scores)
            data.append({"Encoding": enc_type, "Accuracy": avg_acc})

df = pd.DataFrame(data)

# --- Plotting ---
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Encoding", y="Accuracy")
plt.title("Accuracy by Encoding")
plt.ylabel("Overall Accuracy")
plt.tight_layout()
plt.savefig("boxplot_accuracy_by_encoding.png")
plt.show()

