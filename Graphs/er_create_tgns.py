import numpy as np
import pickle

from Graphs.er_step import step_graph, StepConfig
from Graphs.er_graphs import create_er_graph

def er_graph_to_array(graph, timestamp=0):
    return np.array([[u, v, timestamp] for u, v in graph.edges()])

def create_er_tgn(initial_graph, step_config, iterations):
    edges = er_graph_to_array(initial_graph, timestamp=0)

    tgn = [{
        "edges": edges,
        "nodes": set(initial_graph.nodes()),
        "new_nodes": set(),
        "removed_nodes": set(),
        "removed_edges": set(),
        "added_edges": set()
    }]

    for t in range(1, iterations):
        edges, new_nodes, removed_nodes, removed_edges, added_edges = step_graph(edges, step_config, current_time=t)
        tgn.append({
            "edges": edges,
            "nodes": set(edges[:, :2].flatten()),
            "new_nodes": new_nodes,
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "added_edges": added_edges
        })

    return tgn

step_config = StepConfig(n_add=2, p_add=0.5, p_remove=0.5)
iterations = 6

# 4 ER TGNs
for i in range(1, 5):
    print(f"Generating ER graph {i}...")
    edge_prob = 0.3 + 0.1 * i  # vary probability a bit
    er_graph = create_er_graph(num_nodes=5 + i, edge_prob=edge_prob, seed=i)
    tgn = create_er_tgn(er_graph, step_config, iterations)

    filename = f"er{i}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(tgn, f)
    print(f"Saved to {filename}")



