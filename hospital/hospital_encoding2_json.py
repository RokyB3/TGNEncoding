#!/usr/bin/env python3

import os
import pickle
import json

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding2_json.txt"

def convert_to_int(val):
    return int(val) if hasattr(val, "__int__") else val

def convert_set(s):
    return sorted([convert_to_int(n) for n in s]) if s else []

def convert_edges(edges):
    # Ensure edges are converted to list of tuples first (handles NumPy arrays)
    if hasattr(edges, "tolist"):
        edges = edges.tolist()
    return sorted([(convert_to_int(u), convert_to_int(v)) for u, v in edges]) if len(edges) > 0 else []

def build_hybrid_json(tgn):
    """Build structured JSON lines for hybrid encoding"""
    snapshots = []

    # Initial snapshot
    snapshots.append({
        "time": 0,
        "nodes": convert_set(tgn[0]["nodes"]),
        "edges": convert_edges(tgn[0]["edges"])
    })

    # Changes over time
    for t in range(1, len(tgn)):
        step = tgn[t]
        record = {"time": t}

        if step.get("new_nodes"):
            record["added_nodes"] = convert_set(step["new_nodes"])
        if step.get("removed_nodes"):
            record["removed_nodes"] = convert_set(step["removed_nodes"])
        if step.get("added_edges"):
            record["added_edges"] = convert_edges(step["added_edges"])
        if step.get("removed_edges"):
            record["removed_edges"] = convert_edges(step["removed_edges"])

        snapshots.append(record)

    return snapshots

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    hybrid_json_lines = build_hybrid_json(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for entry in hybrid_json_lines:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved hybrid encoding (vertical JSON format) to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    for line in hybrid_json_lines[:3]:
        print(json.dumps(line))

if __name__ == "__main__":
    main()
