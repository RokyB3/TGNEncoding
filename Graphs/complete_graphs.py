import networkx as nx

def create_complete_graph(num_nodes=10):
    """
    Create a complete graph with the given number of nodes.
    """
    g = nx.complete_graph(num_nodes)
    g.graph['name'] = f"Complete_Graph_{num_nodes}"
    return g
