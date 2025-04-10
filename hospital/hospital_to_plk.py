# save as hospital_to_pkl.py
import pandas as pd
import numpy as np
import pickle

# Load and process the hospital contact data
df = pd.read_csv("contacts.dat", sep="\t", header=None, names=["time", "i", "j", "Si", "Sj"])

# Bucket by 120-minute intervals (4*1800 seconds)
df["time_bin"] = df["time"] // (4*1800)

snapshots = []
last_edges = set()
last_nodes = set()

for t_bin, group in df.groupby("time_bin"):
    # Build current edge set (undirected, sorted tuples)
    edges = set(tuple(sorted([row["i"], row["j"]])) for _, row in group.iterrows())
    
    # Extract current node set
    nodes = set([i for edge in edges for i in edge])

    # Track changes
    added_edges = edges - last_edges
    removed_edges = last_edges - edges

    new_nodes = nodes - last_nodes
    removed_nodes = last_nodes - nodes

    snapshot = {
        "edges": np.array(list(edges)),
        "nodes": nodes,
        "new_nodes": new_nodes,
        "removed_nodes": removed_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges
    }

    snapshots.append(snapshot)
    last_edges = edges
    last_nodes = nodes

# Save as hospital.pkl
with open("hospital.pkl", "wb") as f:
    pickle.dump(snapshots, f)

print("Saved hospital.pkl with", len(snapshots), "snapshots")
