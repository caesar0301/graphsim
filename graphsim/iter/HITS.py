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

__all__ = [ 'hits' ]


def normalized(a, axis=0, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


@params(G=nx.DiGraph, max_iter=int, eps=float)
def hits(G, max_iter=100, eps=1e-4):
    """HITS algorithm:
    calculate the hub and authority scores for nodes in a graph.

    Return
    ------
    A 2d matrix of [hub, auth] scores.

    Reference
    ---------
    [1] Kleinberg, Jon M. "Authoritative Sources in a Hyperlinked
        Environment." JACM, 1999.
    """
    N = len(G.nodes())
    A = nx.adjacency_matrix(G).todense()
    Mu = np.concatenate((np.zeros((N, N)), A), 1)
    Md = np.concatenate((A.T, np.zeros((N, N))), 1)
    M = np.concatenate((Mu, Md), 0)

    ha_prev = np.zeros((N*2, 1))
    ha = np.ones((N*2, 1))

    for i in range(max_iter):
        if np.allclose(ha, ha_prev, atol=eps):
            break
        ha_prev = np.copy(ha)
        ha = normalized(np.dot(M, ha_prev))

    print("Converge after %d iterations (eps=%f)." % (i, eps))

    return np.reshape(ha, newshape=(N, 2), order=1)


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(1,4), (1,5), (1,6), (2,5), (2,7), (3,4), (3,5), (3,6), (3,7)])
    print(hits(G))
