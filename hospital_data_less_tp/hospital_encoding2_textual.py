#!/usr/bin/env python3

import os
import pickle
import numpy as np

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding2_textual.txt"

def display_nodes(nodes):
    return ", ".join(str(n) for n in sorted(nodes)) if nodes else "none"

def display_edges(edges):
    return ", ".join(f"{u}–{v}" for u, v in edges) if len(edges) > 0 else "none"

def textual_hybrid_encoding(tgn):
    embedding = "Initial graph (t=0):\n"
    embedding += f"Nodes: {display_nodes(tgn[0]['nodes'])}\n"
    embedding += f"Edges: {display_edges(tgn[0]['edges'])}\n\n"
    embedding += "Changes over time:\n"

    for t in range(1, len(tgn)):
        step = tgn[t]
        if step["new_nodes"]:
            embedding += f"t={t}: Added nodes: {display_nodes(step['new_nodes'])}\n"
        if step["removed_edges"]:
            embedding += f"t={t}: Removed edges: {display_edges(step['removed_edges'])}\n"
        if step["added_edges"]:
            embedding += f"t={t}: Added edges: {display_edges(step['added_edges'])}\n"
        if step["removed_nodes"]:
            embedding += f"t={t}: Removed nodes: {display_nodes(step['removed_nodes'])}\n"

    return embedding

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    embedding = textual_hybrid_encoding(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved textual hybrid encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])

if __name__ == "__main__":
    main()
