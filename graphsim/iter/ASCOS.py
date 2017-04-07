"""
ASCOS similarity measure
"""
#!/usr/bin/env python
# Copyright (C) 2004-2010 by
# Hung-Hsuan Chen <hhchen@psu.edu>
# All rights reserved.
# BSD license.
# NetworkX:http://networkx.lanl.gov/.
__author__ = """Hung-Hsuan Chen (hhchen@psu.edu)"""

import copy
import math
import networkx as nx
import numpy

__all__ = ['ascos']


def ascos(G, c=0.9, max_iter=100, is_weighted=False, remove_neighbors=False, remove_self=False, dump_process=False):
    """Return the ASCOS similarity between nodes

    Parameters
    -----------
    G: graph
      A NetworkX graph
    c: float, 0 < c <= 1
      The number represents the relative importance between in-direct neighbors
      and direct neighbors
    max_iter: integer
      The number specifies the maximum number of iterations for ASCOS
      calculation
    is_weighted: boolean
      Whether use weighted ASCOS or not
    remove_neighbors: boolean
      if true, the similarity value between neighbor nodes is set to zero
    remove_self: boolean
      if true, the similarity value between a node and itself is set to zero
    dump_process: boolean
      if true, the calculation process is dumped

    Returns
    -------
    node_ids : list of node ids
    sim : numpy matrix
      sim[i,j] is the similarity value between node_ids[i] and node_ids[j]

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0,7), (0,1), (0,2), (0,3), (1,4), (2,4), (3,4), (4,5), (4,6)])
    >>> networkx_addon.similarity.ascos(G)

    Notes
    -----

    References
    ----------
    [1] ASCOS: an Asymmetric network Structure COntext Similarity measure.
    Hung-Hsuan Chen and C. Lee Giles.    ASONAM 2013
    """

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("ascos() not defined for graphs with multiedges.")

    if G.is_directed():
        raise Exception("ascos() not defined for directed graphs.")

    node_ids = G.nodes()
    node_id_lookup_tbl = { }
    for i, n in enumerate(node_ids):
        node_id_lookup_tbl[n] = i

    nb_ids = [G.neighbors(n) for n in node_ids]
    nbs = [ ]
    for nb_id in nb_ids:
        nbs.append([node_id_lookup_tbl[n] for n in nb_id])
    del(node_id_lookup_tbl)

    n = G.number_of_nodes()
    sim = numpy.eye(n)
    sim_old = numpy.zeros(shape = (n, n))

    for iter_ctr in range(max_iter):
        if _is_converge(sim, sim_old, n, n):
            break
        sim_old = copy.deepcopy(sim)
        for i in range(n):
            if dump_process:
                print(iter_ctr, ':', i, '/', n)
            for j in range(n):
                if not is_weighted:
                    if i == j:
                        continue
                    s_ij = 0.0
                    for n_i in nbs[i]:
                        s_ij += sim_old[n_i, j]
                    sim[i, j] = c * s_ij / len(nbs[i]) if len(nbs[i]) > 0 else 0
                else:
                    if i == j:
                        continue
                    s_ij = 0.0
                    for n_i in nbs[i]:
                        w_ik = G[node_ids[i]][node_ids[n_i]]['weight'] if 'weight' in G[node_ids[i]][node_ids[n_i]] else 1
                        s_ij += float(w_ik) * (1 - math.exp(-w_ik)) * sim_old[n_i, j]

                    w_i = G.degree(weight='weight')[node_ids[i]]
                    sim[i, j] = c * s_ij / w_i if w_i > 0 else 0

    if remove_self:
        for i in range(n):
            sim[i,i] = 0

    if remove_neighbors:
        for i in range(n):
            for j in nbs[i]:
                sim[i,j] = 0

    return node_ids, sim

def _is_converge(sim, sim_old, nrow, ncol, eps=1e-4):
    for i in range(nrow):
        for j in range(ncol):
            if abs(sim[i,j] - sim_old[i,j]) >= eps:
                return False
    return True
