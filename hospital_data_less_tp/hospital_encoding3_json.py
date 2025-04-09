#!/usr/bin/env python3

import os
import pickle
import json

INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/embeddings/hospital_encoding3_json.txt"

def convert_to_int(val):
    """Convert NumPy int64 or similar to Python int"""
    return int(val) if hasattr(val, '__int__') else val

def build_adjacency_json_lines(tgn):
    adjacency_history = {}

    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]

        # Record additions
        for u, v in edges:
            u, v = convert_to_int(u), convert_to_int(v)
            for src, tgt in [(u, v), (v, u)]:
                if src not in adjacency_history:
                    adjacency_history[src] = {}
                if tgt not in adjacency_history[src]:
                    adjacency_history[src][tgt] = {"added": t, "removed": None}

        # Record removals
        for u, v in snapshot.get("removed_edges", []):
            u, v = convert_to_int(u), convert_to_int(v)
            for src, tgt in [(u, v), (v, u)]:
                if src in adjacency_history and tgt in adjacency_history[src]:
                    adjacency_history[src][tgt]["removed"] = t

    # Convert to compact JSON lines
    lines = []
    for node, neighbors in sorted(adjacency_history.items()):
        connections = []
        for neighbor, times in neighbors.items():
            conn = {
                "neighbor": convert_to_int(neighbor),
                "added": convert_to_int(times["added"])
            }
            if times["removed"] is not None:
                conn["removed"] = convert_to_int(times["removed"])
            connections.append(conn)
        entry = {
            "node": convert_to_int(node),
            "connections": connections
        }
        lines.append(json.dumps(entry, separators=(",", ":")))

    return lines

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    print(f"Loaded {INPUT_FILE} with {len(tgn)} snapshots.")

    lines = build_adjacency_json_lines(tgn)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved horizontal adjacency list JSON to: {OUTPUT_FILE}")
    print("\nPreview:\n")
    print("\n".join(lines[:3]))

if __name__ == "__main__":
    main()
