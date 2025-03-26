import numpy as np

from step import step_graph, StepConfig
from ba_graphs import create_BA_graph

ba_graph = create_BA_graph(5, 2)

print("ba_graph: ", ba_graph)
step_config = StepConfig(n_add=2, p_add=0.5, p_remove=0.5)

edges, new_nodes, removed_nodes, removed_edges, added_edges = step_graph(ba_graph, step_config)

print("edges: ", edges)
print("new_nodes: ", new_nodes)
print("removed_nodes: ", removed_nodes)
print("removed_edges: ", removed_edges)
print("added_edges: ", added_edges)