# Hybrid Encoding (Snapshot + Change Log)
# Hybrid encoding balances both temporal and structural representation, making it ideal for LLM-based reasoning.
import pickle

def display_edges(edges):
    string = ""
    for edge in edges:
        string += f"({edge[0]}–{edge[1]})" + ", "
    return string[:-2] if string else "none"

def display_nodes(nodes):
    return ", ".join(str(n) for n in nodes) if nodes else "none"

def encoding2(tgn):
    embedding = "Initial Graph (t=0):\n"
    initial_nodes = display_nodes(tgn[0]['nodes'])
    initial_edges = display_edges(tgn[0]['edges'])

    embedding += f"Nodes: {initial_nodes}\n"
    embedding += f"Edges: {initial_edges}\n\n"
    embedding += "Temporal Updates:\n"

    for t in range(1, len(tgn)):
        step = tgn[t]
        if step['new_nodes']:
            embedding += f"t={t} → Added Node(s) {display_nodes(step['new_nodes'])}\n"
        if step['removed_edges'].size > 0:
            removed = display_edges(step['removed_edges'])
            embedding += f"t={t} → Removed Edge(s) {removed}\n"
        if step['added_edges'].size > 0:
            added = display_edges(step['added_edges'])
            embedding += f"t={t} → Added Edge(s) {added}\n"
        if step['removed_nodes']:
            embedding += f"t={t} → Node(s) {display_nodes(step['removed_nodes'])} left the graph\n"

    return embedding
