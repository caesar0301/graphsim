"""
SimRank similarity measure.
"""
#!/usr/bin/env python
# Copyright (C) 2015 by
# Xiaming Chen <chen_xm@sjtu.edu.cn>
# All rights reserved.
# BSD license.
import itertools

import numpy as np
import networkx as nx
from typedecorator import params, returns

__author__ = "Xiaming Chen"
__email__ = "chen_xm@sjtu.edu.cn"

__all__ = [ 'simrank', 'simrank_bipartite' ]


@params(G=nx.Graph, r=float, max_iter=int, eps=float)
def simrank(G, r=0.8, max_iter=100, eps=1e-4):
    """ Algorithm of G. Jeh and J. Widom. SimRank: A Measure
    of Structural-Context Similarity. In KDD'02.

    Thanks to Jon Tedesco's answer in SO question #9767773.
    """
    if isinstance(G, nx.MultiGraph):
        assert("The SimRank of MultiGraph is not supported.")

    if isinstance(G, nx.MultiDiGraph):
        assert("The SimRank of MultiDiGraph is not supported.")

    directed = False
    if isinstance(G, nx.DiGraph):
        directed = True

    nodes = G.nodes()
    nodes_i = {}
    for (k, v) in [(nodes[i], i) for i in range(0, len(nodes))]:
        nodes_i[k] = v

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break

        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v: continue

            if directed:
                u_ns, v_ns = G.predecessors(u), G.predecessors(v)
            else:
                u_ns, v_ns = G.neighbors(u), G.neighbors(v)

            # Evaluating the similarity of current nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0:
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))

    print("Converge after %d iterations (eps=%f)." % (i, eps))

    return sim


@params(G=nx.DiGraph, r=float, max_iter=int, eps=float)
def simrank_bipartite(G, r=0.8, max_iter=100, eps=1e-4):
    """ A bipartite version in the paper.
    """
    if not nx.is_bipartite(G):
        assert("A bipartie graph is required.")

    nodes = G.nodes()
    nodes_i = {}
    for (k, v) in [(nodes[i], i) for i in range(0, len(nodes))]:
        nodes_i[k] = v

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    lns = {}
    rns = {}
    for n in nodes:
        preds = G.predecessors(n)
        succs = G.successors(n)
        if len(preds) == 0:
            lns[n] = succs
        else:
            rns[n] = preds

    def _update_partite(ns):
        for u, v in itertools.product(ns.keys(), ns.keys()):
            if u is v: continue
            u_ns, v_ns = ns[u], ns[v]
            if len(u_ns) == 0 or len(v_ns) == 0:
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))

    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        _update_partite(lns)
        _update_partite(rns)

    print("Converge after %d iterations (eps=%f)." % (i, eps))

    return sim


if __name__ == '__main__':
    # Example university web graph in the paper
    G = nx.DiGraph()
    G.add_edges_from([(1,2), (1,3), (2,4), (4,1), (3,5), (5,3)])
    print(simrank(G))

    # Example bipartie graph of cake-bakers in the paper
    G = nx.DiGraph()
    G.add_edges_from([(1,3), (1,4), (1,5), (2,4), (2,5), (2,6)])
    print(simrank_bipartite(G))
