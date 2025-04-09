#!/usr/bin/env python3
"""
hospital_encoding3.py

Generates temporal adjacency list encoding (Encoding 3) for the hospital contact network.
Tracks node-centric edge additions and removals over time.
"""

import os
import pickle
import numpy as np

# Settings
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding3.txt"

# Encoding 3: Temporal Adjacency List
def encoding3(tgn):
    adjacency_history = {}

    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]
        for edge in edges:
            u, v = edge[:2]
            for src, tgt in [(u, v), (v, u)]:  # undirected
                if src not in adjacency_history:
                    adjacency_history[src] = {}
                if tgt not in adjacency_history[src]:
                    adjacency_history[src][tgt] = {"added": t, "removed": None}

        removed_edges = snapshot.get("removed_edges", [])
        for edge in removed_edges:
            u, v = edge[:2]
            for src, tgt in [(u, v), (v, u)]:
                if src in adjacency_history and tgt in adjacency_history[src]:
                    adjacency_history[src][tgt]["removed"] = t

    output = "Temporal Adjacency List:\n"
    for node in sorted(adjacency_history.keys()):
        neighbors = []
        for neighbor in sorted(adjacency_history[node].keys()):
            times = adjacency_history[node][neighbor]
            timeline = f"(t={times['added']}"
            if times["removed"] is not None:
                timeline += f", removed t={times['removed']}"
            timeline += ")"
            neighbors.append(f"{neighbor} {timeline}")
        output += f"{node}: [{', '.join(neighbors)}]\n"

    return output

# Run and Save
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    embedding = encoding3(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved temporal adjacency list encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])  # Preview first 1000 characters

if __name__ == "__main__":
    main()
