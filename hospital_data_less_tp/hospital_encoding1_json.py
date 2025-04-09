#!/usr/bin/env python3
"""
hospital_encoding1_json.py

Generates snapshot-based encoding (Encoding 1) for the hospital contact network.
Stores results as a horizontally structured JSON file, one line per snapshot.
"""

import os
import pickle
import json

# === Settings ===
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding1_json.txt"

# === Main JSON Encoding ===

def build_snapshot_lines(tgn):
    """Generate individual JSON lines per snapshot in compact format"""
    lines = []
    header = {
        "description": "The following is a structured representation of a temporal graph network encoded as a sequence of edges."
    }
    lines.append(json.dumps(header, separators=(",", ":")))

    for t, snapshot in enumerate(tgn):
        edges = snapshot.get("edges", [])
        edge_list = [(int(u), int(v)) for u, v in edges]
        snapshot_entry = {
            "time": t,
            "edges": edge_list
        }
        lines.append(json.dumps(snapshot_entry, separators=(",", ":")))

    return lines

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    lines = build_snapshot_lines(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved horizontal JSON lines to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print("\n".join(lines[:3]))  # Preview first 3 lines

if __name__ == "__main__":
    main()
