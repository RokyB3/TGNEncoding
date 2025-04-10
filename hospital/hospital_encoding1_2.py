#!/usr/bin/env python3

import os
import pickle
import numpy as np

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding1_2.txt"

def textual_snapshot_encoding(tgn):
    embedding = "The following is a description of a temporal graph network encoded as a sequence of edges.\n"
    
    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]
        if len(edges) == 0:
            embedding += f"The graph at time {t} contains no edges.\n"
        else:
            edge_strs = [f"({u} {v})" for u, v in edges]
            edge_list = ", ".join(edge_strs)
            embedding += f"The graph at time {t} contains the following edges: {edge_list}.\n"
    
    return embedding

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    embedding = textual_snapshot_encoding(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved textual snapshot encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])

if __name__ == "__main__":
    main()

