import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Settings
ground_truth_file = "ground_truth_hospital.json"
runs = ["gpt_outputs_run1", "gpt_outputs_run2", "gpt_outputs_run3", "gpt_outputs_run4"]
encodings = [
    "hospital_encoding1.json", "hospital_encoding1_2.json", 
    "hospital_encoding2.json", "hospital_encoding2_1.json", "hospital_encoding2_2.json", 
    "hospital_encoding3.json", "hospital_encoding3_1.json"
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

# --- Scoring ---
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

# --- Group encodings by type ---
encoding_type_map = defaultdict(list)
for enc in encodings:
    if "encoding1" in enc:
        encoding_type_map[1].append(enc)
    elif "encoding2" in enc:
        encoding_type_map[2].append(enc)
    elif "encoding3" in enc:
        encoding_type_map[3].append(enc)

# --- Main Loop ---
sns.set(style="whitegrid")

for question in questions:
    rows = []
    for run in runs:
        for enc_type, files in encoding_type_map.items():
            for filename in files:
                path = os.path.join(run, filename)
                if not os.path.exists(path):
                    continue
                with open(path, "r") as f:
                    try:
                        pred_data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                pred = pred_data.get(question)
                truth = ground_truth.get(question)
                acc = score(pred, truth, question)
                rows.append({"Encoding": enc_type, "Accuracy": acc, "File": filename})

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    csv_name = f"accuracy_encoding_type_{question}.csv"
    df.to_csv(csv_name, index=False)
    
    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Encoding", y="Accuracy")
    plt.title(f"Accuracy by Encoding Type - {question}")
    plt.tight_layout()
    plot_name = f"boxplot_accuracy_encoding_type_{question}.png"
    plt.savefig(plot_name)
    plt.close()
    print(f"[Saved] {csv_name}, {plot_name}")
