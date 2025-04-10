#!/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt

# === Config ===
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/plots/edge_density_over_time.png"

def load_temporal_graph(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_edge_counts(tgn):
    return [len(snapshot.get("edges", [])) for snapshot in tgn]

def plot_edge_density(edge_counts, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(edge_counts)), edge_counts, marker='o', linewidth=1)
    plt.title("Edge Density Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Edges")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    tgn = load_temporal_graph(INPUT_FILE)
    edge_counts = extract_edge_counts(tgn)
    plot_edge_density(edge_counts, OUTPUT_FILE)

if __name__ == "__main__":
    main()
