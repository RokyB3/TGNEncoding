#!/usr/bin/env python3

import os
import pickle
import numpy as np

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding1_textual.txt"

def textual_snapshot_encoding(tgn):
    embedding = "This document summarizes the connections between individuals at each time step.\n\n"
    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]
        if len(edges) == 0:
            embedding += f"Time {t}: No active contacts.\n"
        else:
            connections = [f"{edge[0]}–{edge[1]}" for edge in edges]
            conn_text = ", ".join(connections)
            embedding += f"Time {t}: {conn_text}\n"
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
