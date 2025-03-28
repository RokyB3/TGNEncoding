import networkx as nx
import random

def create_er_graph(num_nodes=10, edge_prob=0.3, seed=None):
    '''
    Creates a single Erdos-Renyi random graph
    num_nodes = number of nodes
    edge_prob = probability for edge creation
    '''
    g = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=seed)
    g.graph['name'] = "ER_Graph"
    return g
