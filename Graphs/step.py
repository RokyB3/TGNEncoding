import numpy as np

class StepConfig:
    def __init__(self, n_add, p_add, p_remove):
        self.n_add = n_add
        self.p_add = p_add
        self.p_remove = p_remove

def step_graph(edges, step_config):
    # edges: list of edges
    # step_config: dictionary of configuration parameters
    # returns: updated edges, new nodes, removed_nodes, removed edges, added edges
    
    # Convert same edges to edge in sorted node order
    # edges = list(set(tuple(sorted(edge)) for edge in edges))

    # Convert edges to numpy array if it's not already
    edges = np.array(edges)
    
    # Find all nodes in the original graph
    if len(edges) > 0:
        original_nodes = set(edges.flatten())
    else:
        original_nodes = set()
    
    # Remove edges based on p_remove
    keep_mask = np.random.random(len(edges)) >= step_config.p_remove
    removed_edges = edges[~keep_mask]
    edges = edges[keep_mask]
    
    # Find all nodes in the graph after edge removal
    if len(edges) > 0:
        remaining_nodes = set(edges.flatten())
        current_nodes = np.max(edges) + 1
    else:
        remaining_nodes = set()
        current_nodes = max(original_nodes) + 1 if original_nodes else 0
    
    # Add new nodes
    new_nodes = range(current_nodes, current_nodes + step_config.n_add)
    
    # Connect new nodes to existing nodes based on p_add
    new_edges = []
    for new_node in new_nodes:
        for existing_node in range(current_nodes):
            if np.random.random() < step_config.p_add:
                # Add edge in both directions (for undirected graph)
                new_edges.append([new_node, existing_node])
    

    # Convert new_edges to numpy array if it's not empty
    added_edges = np.array(new_edges) if new_edges else np.empty((0, 2), dtype=int)
    
    # Combine existing and new edges
    if len(added_edges) > 0:
        edges = np.vstack([edges, added_edges]) if len(edges) > 0 else added_edges

    # Identify completely disconnected nodes
    removed_nodes = original_nodes - set(edges.flatten())

    # Convert edges to undirected graph
    # edges = np.vstack([edges, np.flip(edges, axis=1)])
    
    return edges, new_nodes, removed_nodes, removed_edges, added_edges
    