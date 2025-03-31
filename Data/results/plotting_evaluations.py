import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load evaluation results
with open("TGNEncoding/Data/results/evaluation_summary.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract model type (e.g. "ba" from "ba1")
df["model_type"] = df["graph"].str.extract(r"([a-zA-Z]+)")
df["graph_id"] = df["graph"]

# --- 1. Accuracy by Encoding ---
plt.figure(figsize=(8, 6))
sns.boxplot(x="encoding", y="overall_accuracy", data=df)
plt.title("Accuracy by Encoding")
plt.ylabel("Overall Accuracy")
plt.xlabel("Encoding")
plt.tight_layout()
plt.savefig("accuracy_by_encoding.png")
plt.close()

# --- 2. Accuracy by Model Type ---
plt.figure(figsize=(8, 6))
sns.boxplot(x="model_type", y="overall_accuracy", data=df)
plt.title("Accuracy by Model Type")
plt.ylabel("Overall Accuracy")
plt.xlabel("Model Type")
plt.tight_layout()
plt.savefig("accuracy_by_model_type.png")
plt.close()

# --- 3. Accuracy by Graph (sorted) ---
graph_sorted = (
    df.groupby("graph_id")["overall_accuracy"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(x="graph_id", y="overall_accuracy", data=graph_sorted, ci=None)
plt.xticks(rotation=45)
plt.title("Accuracy by Graph (Sorted)")
plt.ylabel("Overall Accuracy")
plt.xlabel("Graph")
plt.tight_layout()
plt.savefig("accuracy_by_graph.png")
plt.close()

# --- 4. Summary Table ---
summary = (
    df.groupby(["model_type", "encoding"])
    .agg(avg_accuracy=("overall_accuracy", "mean"), count=("overall_accuracy", "size"))
    .reset_index()
)
summary.to_csv("summary_table.csv", index=False)

print("Plots and table saved as:")
print(" - accuracy_by_encoding.png")
print(" - accuracy_by_model_type.png")
print(" - accuracy_by_graph.png")
print(" - summary_table.csv")
