import random
from matplotlib import container
import networkx as nx
import matplotlib.pyplot as plt
import time

from itertools import combinations, groupby

def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               draw: bool = False):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    weight = []

    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0,10)
        weight.append(w['weight'])
                
    if draw: 
        plt.figure(figsize=(10,6))
        nx.draw(G, node_color='lightblue', 
            with_labels=True, 
            node_size=500)
    
    return G, weight



def kruskal(graph: list):
    indx_v1 = 0
    indx_v2 = 0
    G = graph[0]
    edges = list(G.edges())
    # edges = [(0, 3), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), 
    # (0, 8), (0, 9), (0, 10), (0, 11), (1, 4), (1, 2), (1, 3), (1, 5), 
    # (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 5), (2, 3), 
    # (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 6), 
    # (3, 4), (3, 5), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 6), 
    # (4, 5), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 6), (5, 7), 
    # (5, 8), (5, 9), (5, 10), (5, 11), (6, 9), (6, 7), (6, 8), (6, 10), 
    # (6, 11), (7, 9), (7, 8), (7, 10), (7, 11), (8, 10), (8, 9), (8, 11), 
    # (9, 11), (9, 10), (10, 11)]
    nodes = list(G.nodes())
    # nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    weights = graph[1]
    # weights = [10, 2, 8, 2, 6, 6, 8, 2, 0, 
    # 0, 1, 3, 5, 5, 5, 4, 5, 8, 7, 1, 3, 10, 0, 
    # 2, 0, 3, 9, 5, 0, 3, 4, 8, 2, 6, 2, 6, 0, 4, 
    # 1, 0, 8, 10, 3, 5, 4, 2, 4, 4, 9, 6, 10, 0, 
    # 0, 5, 3, 1, 1, 1, 10, 10, 10, 3, 4, 8, 1, 1]
    # print(edges)
    # print(weights)
    # print(nodes)
    container_for_union = []
    lst_edges_tree = []
    graph = list(zip(edges, weights))
    graph = sorted(graph, key = lambda x: x[-1])


    # list_of_vertexes = list(set(str(node)) for node in nodes)
    list_of_vertexes = []
    st_container = set()
    for node in nodes:
        st_container.add(str(node))
        list_of_vertexes.append(st_container)
        st_container = set()
    for i in range(0, len(graph)):
        graph[i] = tuple(str(y) for  y in graph[i][0])
        v1 = graph[i][0]
        v2 = graph[i][1]
        for v in range(len(list_of_vertexes)):
            if v1 in list_of_vertexes[v] and v2 in list_of_vertexes[v]:
                break
            if v1 in list_of_vertexes[v]:
                indx_v1 = v
                container_for_union.append(list_of_vertexes[indx_v1])
            if v2 in list_of_vertexes[v]:
                indx_v2 = v
                container_for_union.append(list_of_vertexes[indx_v2])
            if len(container_for_union) == 2:
                good_edge = tuple(int(vertex) for vertex in graph[i])
                lst_edges_tree.append(good_edge)
                to_add = []
                to_add.append(container_for_union[0].union(container_for_union[1]))
                list_of_vertexes.append(to_add[0])
                container_for_union = []
                if indx_v1 > indx_v2:
                    list_of_vertexes.pop(indx_v1)
                    list_of_vertexes.pop(indx_v2)
                else:
                    list_of_vertexes.pop(indx_v2)
                    list_of_vertexes.pop(indx_v1)
                break

    return lst_edges_tree

if __name__== "__main__":
    time_taken = 0
    for i in range(10):
        start = time.time()
        kruskal(gnp_random_connected_graph(500, 1))
        end = time.time()
        time_taken += (end-start)
    print(time_taken)
    print(time_taken/10)
