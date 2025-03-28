import networkx as nx
import random

def create_er_graphs(num_graphs=5, num_nodes=10, edge_prob=0.3, seed=None):
    '''
    Creates a list of Erdos-Renyi random graphs
    num_graphs = number o fgraphs
    num_nodes = number of nodes
    edge_prob = probability for edge creation
    '''
    graphs = []
    for i in range(num_graphs):
        g = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=seed+i if seed is not None else None)
        g.graph['name'] = f"ER_Graph_{i}"
        graphs.append(g)
    return graphs
