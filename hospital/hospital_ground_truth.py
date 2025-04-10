import pickle
import os
import json

# === Config ===
INPUT_FILE = "hospital.pkl"
OUTPUT_FILE = "Data/results/ground_truth_hospital.json"
NODE_QUERY = 1100  # Example node to track for appearance (can adjust)

# === Ground Truth Functions ===

def get_node_first_appearance(tgn, node):
    for t, snapshot in enumerate(tgn):
        if node in snapshot["nodes"]:
            return t
    return None

def time_steps_most_connected(tgn):
    max_connections = 0
    best_times = []
    for t, snapshot in enumerate(tgn):
        edge_count = len(snapshot["edges"])
        if edge_count > max_connections:
            max_connections = edge_count
            best_times = [t]
        elif edge_count == max_connections:
            best_times.append(t)
    return best_times

def time_steps_least_connected(tgn):
    min_connections = float('inf')
    worst_times = []
    for t, snapshot in enumerate(tgn):
        edge_count = len(snapshot["edges"])
        if edge_count < min_connections:
            min_connections = edge_count
            worst_times = [t]
        elif edge_count == min_connections:
            worst_times.append(t)
    return worst_times

def nodes_with_most_edge_changes(tgn):
    edge_changes = {}
    for snapshot in tgn:
        for edge in snapshot["added_edges"]:
            for node in edge[:2]:
                edge_changes[node] = edge_changes.get(node, 0) + 1
        for edge in snapshot["removed_edges"]:
            for node in edge[:2]:
                edge_changes[node] = edge_changes.get(node, 0) + 1
    if not edge_changes:
        return []
    max_changes = max(edge_changes.values())
    return [(node, count) for node, count in edge_changes.items() if count == max_changes]

def nodes_deleted_and_reappeared(tgn):
    node_status = {}  # node → list of (appeared, disappeared) tuples
    current_nodes = set()

    for t, snapshot in enumerate(tgn):
        now_nodes = snapshot["nodes"]

        # newly appeared
        for node in now_nodes - current_nodes:
            if node not in node_status:
                node_status[node] = []
            node_status[node].append({"appeared": t})

        # disappeared
        for node in current_nodes - now_nodes:
            if node in node_status and "appeared" in node_status[node][-1] and "disappeared" not in node_status[node][-1]:
                node_status[node][-1]["disappeared"] = t

        current_nodes = now_nodes

    # Filter: nodes that disappeared and then reappeared later
    reappeared_nodes = []
    for node, events in node_status.items():
        if len(events) > 1:
            reappeared_nodes.append(node)

    return sorted(reappeared_nodes)

# === Main ===

def safe_int(val):
    return int(val) if hasattr(val, '__int__') else val

def safe_pair_list(pairs):
    return [[safe_int(a), safe_int(b)] for a, b in pairs]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        tgn = pickle.load(f)

    results = {
        "node_first_appearance": safe_int(get_node_first_appearance(tgn, NODE_QUERY)),
        "time_steps_most_connected": [safe_int(x) for x in time_steps_most_connected(tgn)],
        "time_steps_least_connected": [safe_int(x) for x in time_steps_least_connected(tgn)],
        "nodes_with_most_edge_changes": safe_pair_list(nodes_with_most_edge_changes(tgn)),
        "nodes_deleted_and_reappeared": nodes_deleted_and_reappeared(tgn),
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ground truth saved to {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
