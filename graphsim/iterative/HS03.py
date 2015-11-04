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

__all__ = ['nsim_hs03']


@params(G1=nx.DiGraph, G2=nx.DiGraph, max_iter=int, eps=float)
def nsim_hs03(G1, G2, max_iter=100, eps=1e-4):
    """
    References
    ----------
    [1] Heymans, M., Singh, A. Deriving phylogenetic trees from the similarity
        analysis of metabolic pathways. Bioinformatics, 2003
    """
    N = len(G1.nodes())
    M = len(G2.nodes())
    A = nx.adjacency_matrix(G1).todense()
    B = nx.adjacency_matrix(G2).todense()
    Ia = np.ones((N, N))
    Ib = np.ones((M, M))

    nsim_prev = np.zeros((M, N))
    nsim = np.ones((M, N))

    for i in range(max_iter):
        if np.allclose(nsim, nsim_prev, atol=eps):
            break

        nsim_prev = np.copy(nsim)
        nsim = \
            np.dot(np.dot(B, nsim_prev), A.T) + \
            np.dot(np.dot(B.T, nsim_prev), A) + \
            np.dot(np.dot((Ib-B), nsim_prev), (Ia-A).T) + \
            np.dot(np.dot((Ib-B).T, nsim_prev), (Ia-A)) - \
            np.dot(np.dot(B, nsim_prev), (Ia-A).T) - \
            np.dot(np.dot(B.T, nsim_prev), (Ia-A)) - \
            np.dot(np.dot((Ib-B), nsim_prev), A.T) - \
            np.dot(np.dot((Ib-B).T, nsim_prev), A)

        fnorm = np.linalg.norm(nsim, ord='fro')
        nsim = nsim / fnorm

    print("Converge after %d iterations (eps=%f)." % (i, eps))

    return nsim.T

if __name__ == '__main__':
    # Example of Fig. 1.2 in paper BVD04.
    G1 = nx.DiGraph()
    G1.add_edges_from([(1,2), (2,1), (1,3), (4,1), (2,3), (3,2), (4,3)])

    G2 = nx.DiGraph()
    G2.add_edges_from([(1,4), (1,3), (3,1), (6,1), (6,4), (6,3), (3,6), (2,4), (2,6), (3,5)])
    nsim = nsim_hs03(G1, G2)
    print(nsim)
