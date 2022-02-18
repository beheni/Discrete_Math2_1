import random
import time

import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations, groupby


def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               draw: bool = False) -> list[tuple[int, int]]:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """

    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(0, 10)

    if draw:
        plt.figure(figsize=(10, 6))
        nx.draw(G, node_color='lightblue',
                with_labels=True,
                node_size=500)
    return G

def step(g, not_chosen_edges=None, not_chosen_v=None, chosen_v=None, chosen_edges=None):
    possible_edges = {}
    if chosen_edges is None:
        chosen_edges = set()
    if chosen_v is None:
        chosen_v = {0}
    if not_chosen_v is None:
        not_chosen_v = set(node for node in g.nodes())
        not_chosen_v.remove(0)
    if not_chosen_edges is None:
        not_chosen_edges = {}
        for u, v, w in g.edges(data=True):
            not_chosen_edges[u, v] = w['weight']
    for tup in not_chosen_edges:
        if tup[0] in chosen_v or tup[1] in chosen_v:
            if tup[0] in not_chosen_v or tup[1] in not_chosen_v:
                if not_chosen_edges[tup] not in possible_edges.keys():
                    possible_edges[not_chosen_edges[tup]] = [tup]
                else:
                    possible_edges[not_chosen_edges[tup]].append(tup)
    if len(possible_edges.keys()) > 0:
        our_choice = min(possible_edges.keys())
        v_1 = possible_edges[our_choice][0][0]
        v_2 = possible_edges[our_choice][0][1]
        chosen_edges.add(tuple([v_1, v_2]))
        chosen_v.add(v_1)
        chosen_v.add(v_2)
        if v_1 in not_chosen_v:
            not_chosen_v.remove(v_1)
        if v_2 in not_chosen_v:
            not_chosen_v.remove(v_2)
        not_chosen_edges.pop(possible_edges[our_choice][0])
    else:
        not_chosen_v = None
    if not_chosen_v != None:
        step(g, not_chosen_edges, not_chosen_v, chosen_v, chosen_edges)
    return chosen_edges


if __name__ == '__main__':
    # time_taken = 0
    # for i in range(10):
    #     start = time.time()
    #     step(gnp_random_connected_graph(500, 1))
    #     end = time.time()
    #     time_taken += (end-start)
    # print(time_taken)
    # print(time_taken/10)
    G = gnp_random_connected_graph(8, 1)
    print(step(G))