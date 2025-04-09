#!/usr/bin/env python3
"""
hospital_encoding2.py

Generates hybrid encoding (Encoding 2) for the hospital contact network.
Includes initial graph snapshot and time-based change logs.
"""

import os
import pickle
import numpy as np

# Settings
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding2.txt"

# Formatting Functions
def display_edges(edges):
    return ", ".join(f"({edge[0]}–{edge[1]})" for edge in edges) if len(edges) > 0 else "none"

def display_nodes(nodes):
    return ", ".join(str(n) for n in sorted(nodes)) if nodes else "none"

# Hybrid Encoding Function
def encoding2(tgn):
    embedding = "Initial Graph (t=0):\n"
    embedding += f"Nodes: {display_nodes(tgn[0]['nodes'])}\n"
    embedding += f"Edges: {display_edges(tgn[0]['edges'])}\n\n"
    embedding += "Temporal Updates:\n"

    for t in range(1, len(tgn)):
        step = tgn[t]
        if step['new_nodes']:
            embedding += f"t={t} → Added Node(s) {display_nodes(step['new_nodes'])}\n"
        if len(step['removed_edges']) > 0:
            embedding += f"t={t} → Removed Edge(s) {display_edges(step['removed_edges'])}\n"
        if len(step['added_edges']) > 0:
            embedding += f"t={t} → Added Edge(s) {display_edges(step['added_edges'])}\n"
        if step['removed_nodes']:
            embedding += f"t={t} → Node(s) {display_nodes(step['removed_nodes'])} left the graph\n"

    return embedding

# Run and Save
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    embedding = encoding2(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved hybrid encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])  # Preview first 1000 characters

if __name__ == "__main__":
    main()
