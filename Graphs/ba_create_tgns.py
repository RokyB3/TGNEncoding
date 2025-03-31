import numpy as np
import pickle
import random

from Graphs.ba_step import step_graph, StepConfig
from Graphs.ba_graphs import create_BA_graph

def create_tgn(ba_graph, step_config, iterations):
    tgn = [{
        "edges": ba_graph,
        "nodes": set(ba_graph.flatten()),
        "new_nodes": set(),
        "removed_nodes": set(),
        "removed_edges": set(),
        "added_edges": set()
    }]

    for _ in range(iterations - 1):
        edges, new_nodes, removed_nodes, removed_edges, added_edges = step_graph(ba_graph, step_config)
        tgn.append({
            "edges": edges,
            "nodes": set(edges.flatten()),
            "new_nodes": new_nodes,
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "added_edges": added_edges
        })
        ba_graph = edges
    return tgn


step_config = StepConfig(n_add=2, p_add=0.5, p_remove=0.5)
iterations = 6

# 4 BA temporal graphs with different seeds
for i in range(1, 5):
    np.random.seed(i)
    random.seed(i)
    ba_graph = create_BA_graph(5, 2)
    tgn = create_tgn(ba_graph, step_config, iterations)

    filename = f"ba{i}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(tgn, f)
    print(f"Saved {filename}")
