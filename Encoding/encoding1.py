import pickle

with open("ba1.pkl", "rb") as f:
    tgn = pickle.load(f)

def display_edges(edges):
    string = ""
    for edge in edges:
        string += str(edge) + ", "
    string = string[:-2]
    return string

def encoding1(tgn):
    embedding = "The following is a description of a temporal graph network encoded as a sequence of edges.\n"
    for i in range(len(tgn)):
        embedding += "The graph at time " + str(i) + " contains the following edges: " + display_edges(tgn[i]["edges"]) + ".\n"
    return embedding

print(encoding1(tgn))