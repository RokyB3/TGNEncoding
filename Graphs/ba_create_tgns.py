import numpy as np
import pickle

from Graphs.ba_step import step_graph, StepConfig
from Graphs.ba_graphs import create_BA_graph

ba_graph = create_BA_graph(5, 2)

step_config = StepConfig(n_add=2, p_add=0.5, p_remove=0.5)

def create_tgn(ba_graph, step_config, iterations):
    tgn = [{
        "edges": ba_graph,
        "nodes": set(ba_graph.flatten()),
        "new_nodes": set(),
        "removed_nodes": set(),
        "removed_edges": set(),
        "added_edges": set()
    }]

    for _ in range(iterations-1):
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


tgn = create_tgn(ba_graph, step_config, 6)

with open("ba1.pkl", "wb") as f:
    pickle.dump(tgn, f)

with open("ba1.pkl", "rb") as f:
    tgn = pickle.load(f)

print("tgn: ", tgn)