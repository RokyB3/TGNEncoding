import numpy as np
import math

def create_BA_graph(n, m):
    # n: Number of nodes
    # m: Average degree
    sum_k = 0
    k_count = np.zeros(n, dtype=int)
    k_dist = np.zeros(n, dtype=float)
    edges = []
    for i in range(n):
        if i == 0:
            k_count[i] = 0
        elif i < m:
            sum_k += 2*(i)
            k_count[i] = i
            for j in range(i):
                edges.append([i, j])
                edges.append([j, i])
                k_count[j] += 1
        else:
            # Compute k_dist
            k_dist = (k_count / sum_k)
            # Check how many nodes to choose
            prob = m-math.floor(m)
            edges_to_add = np.random.choice([math.floor(m), math.ceil(m)], p=[1-prob, prob])
            # Choose nodes to connect to
            chosen_nodes = np.random.choice(range(i), size=edges_to_add, replace=False, p=k_dist[:i])
            # Add edges
            for node in chosen_nodes:
                edges.append([i, node])
                edges.append([node, i])
                k_count[node] += 1
            k_count[i] = edges_to_add
            sum_k += 2*edges_to_add
    return np.array(edges)