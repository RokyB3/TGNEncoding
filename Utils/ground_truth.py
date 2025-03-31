import pickle

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


