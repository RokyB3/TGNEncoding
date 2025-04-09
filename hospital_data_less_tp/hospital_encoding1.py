#!/usr/bin/env python3
"""
hospital_encoding1.py

Generates snapshot-based encoding (Encoding 1) for the hospital contact network.
"""

import os
import pickle
import numpy as np

# ==== Settings ====
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding1.txt"

# ==== Snapshot-Based Encoding Function ====

def display_edges(edges):
    """Format edges like [u v], [x y], ..."""
    return ", ".join(f"[{edge[0]} {edge[1]}]" for edge in edges)

def encoding1(tgn):
    """Generate snapshot-style text encoding from TGN list."""
    embedding = "The following is a description of a temporal graph network encoded as a sequence of edges.\n"
    for i, snapshot in enumerate(tgn):
        edges = snapshot.get("edges", [])
        embedding += f"The graph at time {i} contains the following edges: {display_edges(edges)}.\n"
    return embedding

# ==== Run and Save ====

def main():
    # Load hospital temporal graph
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    # Encode
    embedding = encoding1(tgn)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save encoding
    with open(OUTPUT_FILE, "w") as f:
        f.write(embedding)

    print(f"Saved hospital snapshot encoding to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(embedding[:1000])  # First 1000 chars preview

if __name__ == "__main__":
    main()

