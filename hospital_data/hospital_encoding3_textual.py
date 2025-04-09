#!/usr/bin/env python3

import os
import pickle
import numpy as np

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding3_textual.txt"

def textual_adjacency_encoding(tgn):
    adjacency_history = {}

    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]
        for u, v in edges:
            for src, tgt in [(u, v), (v, u)]:
                if src not in adjacency_history:
                    adjacency_history[src] = {}
                if tgt not in adjacency_history[src]:
                    adjacency_history[src][tgt] = {"added": t, "removed": None}

        for u, v in snapshot.get("removed_edges", []):
            for src, tgt in [(u, v), (v, u)]:
                if src in adjacency_history and tgt in adjacency_history[src]:
                    adjacency_history[src][tgt]["removed"] = t

    output = "Each node's connections over time:\n\n"
    for node in sorted(adjacency_history.keys()):
        output += f"Node {node}:\n"
        for neighbor in sorted(adjacency_history[node].keys()):
            times = adjacency_history[node][neighbor]
            line = f"  - Connected to {neighbor} at t={times['added']}"
            if times["removed"] is not None:
                line += f", disconnected at t={times['removed']}"
            output += line + "\n"
        output += "\n"

    return output

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    embedding = textual_adjacency_encoding(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved textual adjacency list encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])

if __name__ == "__main__":
    main()
