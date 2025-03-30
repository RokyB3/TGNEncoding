# Temporal Adjacency List (For Explicit Structural Tracking Over Time)
# This method keeps node-centric temporal changes while tracking evolving edges in a compact way.

def encoding3(tgn):
    adjacency_history = {}
    for t, snapshot in enumerate(tgn):
        edges = snapshot["edges"]
        for edge in edges:
            u, v = edge[:2]
            for src, tgt in [(u, v), (v, u)]:  # undirected
                if src not in adjacency_history:
                    adjacency_history[src] = {}
                if tgt not in adjacency_history[src]:
                    adjacency_history[src][tgt] = {"added": t, "removed": None}
    
        removed_edges = snapshot.get("removed_edges", [])
        for edge in removed_edges:
            u, v = edge[:2]
            for src, tgt in [(u, v), (v, u)]:
                if src in adjacency_history and tgt in adjacency_history[src]:
                    adjacency_history[src][tgt]["removed"] = t

    output = "Temporal Adjacency List:\n"
    for node in sorted(adjacency_history.keys()):
        neighbors = []
        for neighbor in sorted(adjacency_history[node].keys()):
            times = adjacency_history[node][neighbor]
            timeline = f"(t={times['added']}"
            if times["removed"] is not None:
                timeline += f", removed t={times['removed']}"
            timeline += ")"
            neighbors.append(f"{neighbor} {timeline}")
        output += f"{node}: [{', '.join(neighbors)}]\n"
    return output