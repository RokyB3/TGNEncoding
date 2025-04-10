
import matplotlib.pyplot as plt
import os
import re

# File paths (update if needed)
paths = {
    "encoding1": "Data/embeddings/hospital_encoding1.txt",
    "encoding2_1": "Data/embeddings/hospital_encoding2_1.txt",
    "encoding2": "Data/embeddings/hospital_encoding2.txt",
    "encoding3": "Data/embeddings/hospital_encoding3.txt"
}

# Time steps to analyze
# Time steps to analyze
time_steps = list(range(0, 45))

# Function to extract edge count from encoding1
def parse_encoding1(path):
    edge_counts = {}
    with open(path, "r") as f:
        for line in f:
            match = re.match(r"The graph at time (\d+) contains the following edges: (.*)\.", line.strip())
            if match:
                t = int(match.group(1))
                edges = match.group(2)
                edge_counts[t] = len(edges.split(", ")) if edges else 0
    return edge_counts

# Function to reconstruct edge count per snapshot from encoding2_1
def parse_encoding2_1(path):
    current_edges = set()
    edge_counts = {}
    t = 0  # Start at time 0

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Initial graph") or line.startswith("Nodes:"):
            continue

        if line.startswith("t="):
            match = re.search(r"t=(\d+)", line)
            if match:
                t = int(match.group(1))

        elif line.startswith("Edges:"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.add((int(u), int(v)))
            edge_counts[t] = len(current_edges)  # <-- Use dynamic t here

        elif line.startswith("Removed edges"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.discard((int(u), int(v)))

        elif line.startswith("Added edges"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.add((int(u), int(v)))
            edge_counts[t] = len(current_edges)  # <-- Correctly store for this t

    return edge_counts






# Function to reconstruct from encoding2 (supports t=1 → format)
def parse_encoding2(path):
    edge_counts = {}
    current_edges = set()
    with open(path, "r") as f:
        lines = f.readlines()

    t = 0
    for line in lines:
        if line.startswith("Initial graph") or line.startswith("Nodes:"):
            continue
        elif line.startswith("Edges:"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.add((int(u), int(v)))
            edge_counts[0] = len(current_edges)
        elif line.startswith("t="):
            t_match = re.search(r"t=(\d+)", line)
            if t_match:
                t = int(t_match.group(1))
        elif line.startswith("Removed edges"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.discard((int(u), int(v)))
        elif line.startswith("Added edges"):
            edges = re.findall(r"(\d+)[\u2013\u2014\-](\d+)", line)
            for u, v in edges:
                current_edges.add((int(u), int(v)))
            edge_counts[t] = len(current_edges)
    return edge_counts

# Function to estimate edge count from encoding3 adjacency lists
def parse_encoding3(path):
    edge_counts = {t: 0 for t in time_steps}
    with open(path, "r") as f:
        content = f.read()

    current_node = None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Node"):
            current_node = int(line.split()[1][:-1])
        elif line.startswith("- Connected to"):
            parts = re.findall(r"(\d+)", line)
            if len(parts) >= 2:
                neighbor = int(parts[0])
                added = int(parts[1])
                removed = int(parts[2]) if len(parts) > 2 else None
                for t in time_steps:
                    if t >= added and (removed is None or t < removed):
                        edge_counts[t] += 1

    for t in edge_counts:
        edge_counts[t] //= 2  # undirected
    return edge_counts

# Parse all encodings
#enc1_counts = parse_encoding1(paths["encoding1"])
#enc2_1_counts = parse_encoding2_1(paths["encoding2_1"])
#enc2_counts = parse_encoding2(paths["encoding2"])
enc3_counts = parse_encoding3(paths["encoding3"])

# Plot
plt.figure(figsize=(12, 6))
for label, counts in [
    #("Encoding1", enc1_counts),
    #("Encoding2_1", enc2_1_counts)
    #("Encoding2", enc2_counts),
    ("Encoding3", enc3_counts)
]:
    y = [counts.get(t, 0) for t in time_steps]
    plt.plot(time_steps, y, marker='o', label=label)

plt.title("Edge Counts from Time Steps 0 to 44")
plt.xlabel("Time Step")
plt.ylabel("Edge Count")
plt.xticks(time_steps)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()