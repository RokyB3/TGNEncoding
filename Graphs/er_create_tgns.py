import numpy as np
import pickle

from er_step import step_graph, StepConfig
from er_graphs import create_er_graphs

# Create initial ER graph (as edge list with 3 columns)
def er_graph_to_array(graph, timestamp=0):
    return np.array([[u, v, timestamp] for u, v in graph.edges()])

# Create TGN from ER graph
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


er_graph = create_er_graphs(num_graphs=1, num_nodes=5, edge_prob=0.4)[0]

step_config = StepConfig(n_add=2, p_add=0.5, p_remove=0.5)
tgn = create_er_tgn(er_graph, step_config, iterations=3)

with open("er1.pkl", "wb") as f:
    pickle.dump(tgn, f)

print("tgn: ", tgn)


