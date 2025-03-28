import numpy as np

class StepConfig:
    def __init__(self, n_add, p_add, p_remove):
        self.n_add = n_add
        self.p_add = p_add
        self.p_remove = p_remove

def step_graph(edges, step_config, current_time=1):
    edges = np.array(edges)

    # track existing nodes
    original_nodes = set(edges[:, :2].flatten()) if len(edges) > 0 else set()

    # remove edges based on probability
    keep_mask = np.random.random(len(edges)) >= step_config.p_remove
    removed_edges = edges[~keep_mask]
    edges = edges[keep_mask]

    # update node range
    if len(edges) > 0:
        current_nodes = int(np.max(edges[:, :2])) + 1
    else:
        current_nodes = max(original_nodes) + 1 if original_nodes else 0

    # add new nodes
    new_nodes = list(range(current_nodes, current_nodes + step_config.n_add))

    # new edges
    new_edges = []
    for new_node in new_nodes:
        for existing_node in range(current_nodes):
            if np.random.random() < step_config.p_add:
                new_edges.append([new_node, existing_node])

    added_edges = np.array(new_edges) if new_edges else np.empty((0, 2), dtype=int)

    # timestamps
    if added_edges.shape[1] == 2:
        added_edges = np.hstack([added_edges, np.full((added_edges.shape[0], 1), current_time)])
    if removed_edges.shape[1] == 2:
        removed_edges = np.hstack([removed_edges, np.full((removed_edges.shape[0], 1), current_time)])

    # combine old and new edges
    if len(added_edges) > 0:
        edges = np.vstack([edges, added_edges]) if len(edges) > 0 else added_edges

    removed_nodes = original_nodes - set(edges[:, :2].flatten())

    return edges, new_nodes, removed_nodes, removed_edges, added_edges
