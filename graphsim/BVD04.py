import itertools

import numpy as np
import networkx as nx
from typedecorator import params, returns


__all__ = [ 'nsim_matrix_bvd04' ]


@params(G1=nx.DiGraph, G2=nx.DiGraph, max_iter=int, eps=float)
def nsim_matrix_bvd04(G1, G2, max_iter=100, eps=1e-4):
    """
    Algorithm to calculate node-node similarity matrix of
    two directed graphs.

    Return
    ------
    A 2d similarity matrix of |V1| x |V2|.

    Reference
    ---------
    Blondel, Vincent D. et al. "A Measure of Similarity between Graph Vertices:
    Applications to Synonym Extraction and Web Searching." SIAM Review (2004)
    """
    N = len(G1.nodes())
    M = len(G2.nodes())
    A = nx.adjacency_matrix(G1).todense()
    B = nx.adjacency_matrix(G2).todense()
    nsim_prev = np.zeros((M, N))
    nsim = np.ones((M, N))

    for i in range(max_iter):
        if np.allclose(nsim, nsim_prev, atol=eps):
            break

        nsim_prev = np.copy(nsim)
        nsim = np.dot(np.dot(B, nsim_prev), A.T) + \
            np.dot(np.dot(B.T, nsim_prev), A)

        fnorm = np.linalg.norm(nsim, ord='fro')
        nsim = nsim / fnorm

    return nsim.T


if __name__ == '__main__':
    # Example in BVD04 Fig. 1.2
    G1 = nx.DiGraph()
    G1.add_edges_from([(1,2), (2,1), (1,3), (4,1), (2,3), (3,2), (4,3)])

    G2 = nx.DiGraph()
    G2.add_edges_from([(1,4), (1,3), (3,1), (6,1), (6,4), (6,3), (3,6), (2,4), (2,6), (3,5)])
    print nsim_matrix_bvd04(G1, G2)
