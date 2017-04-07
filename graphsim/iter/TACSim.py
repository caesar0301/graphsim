"""
Topology-Attributes Coupling Simmilarity (TACSim) measure.
"""
#!/usr/bin/env python
# Copyright (C) 2015 by
# Xiaming Chen <chen_xm@sjtu.edu.cn>
# All rights reserved.
# BSD license.
import itertools, copy

import numpy as np
import networkx as nx
from typedecorator import params

__author__ = "Xiaming Chen"
__email__ = "chen_xm@sjtu.edu.cn"

__all__ = [ 'tacsim', 'tacsim_combined', 'normalized', 'node_edge_adjacency' ]


def _strength_nodes(nw1, nw2, ew):
    return 1.0 * nw1 * nw2 / np.power(ew, 2)


def _strength_edges(ew1, ew2, nw):
    return 1.0 * np.power(nw, 2) / (ew1 * ew2)


def _coherence(s1, s2):
    return 2.0 * np.sqrt(s1 * s2) / (s1 + s2)


def _converged(nsim, nsim_prev, esim, esim_prev, eps=1e-4):
    if np.allclose(nsim, nsim_prev, atol=eps) and \
            np.allclose(esim, esim_prev, atol=eps):
        return True
    return False


def normalized(a, axis=None, order=None):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / l2


def _mask_lower_values(m, tol=1e-6):
    m[abs(m) < tol] = 0.0
    return m


def _graph_elements(G, node_attribute='weight', edge_attribute='weight', dummy_eps=1e-3):
    """ Generate strength matrices and node-edge indexes mapping of nodes and edges.
    """
    nodes = G.nodes()
    edges = G.edges()
    V = len(nodes)
    E = len(edges)
    node_id_lookup_tbl = {}
    edge_id_lookup_tbl = {}
    node_weight_vec = np.ones(V)
    edge_weight_vec = np.ones(E)
    node_strength_mat = np.empty((V, V)); node_strength_mat.fill(-1)
    edge_strength_mat = np.empty((E, E)); edge_strength_mat.fill(-1)
    node_edge_map = {}
    edge_node_map = {}

    for i, n in enumerate(nodes):
        node_id_lookup_tbl[n] = i
        try:
            node_weight_vec[i] = G.node[n][node_attribute]
            if node_weight_vec[i] < 0: # to fix dummy nodes
                node_weight_vec[i] = dummy_eps
        except KeyError:
            node_weight_vec[i] = 1

    node_weight_vec = normalized(node_weight_vec)

    for i, e in enumerate(edges):
        edge_id_lookup_tbl[e] = i
        try:
            edge_weight_vec[i] = G.edge[e[0]][e[1]][edge_attribute]
            if edge_weight_vec[i] <= 0: # to fix dummy edges
                edge_weight_vec[i] = dummy_eps
        except KeyError:
            edge_weight_vec[i] = 1

    edge_weight_vec = normalized(edge_weight_vec)

    for e in edges:
        n0, n1 = node_id_lookup_tbl[e[0]], node_id_lookup_tbl[e[1]]
        e01 = edge_id_lookup_tbl[e]
        node_strength_mat[n0][n1] = \
            _strength_nodes(node_weight_vec[n0], node_weight_vec[n1], edge_weight_vec[e01])
        # record node-node intersection
        if n0 not in node_edge_map:
            node_edge_map[n0] = {}
        node_edge_map[n0][n1] = e01

    for n in nodes:
        n01 = node_id_lookup_tbl[n]
        preds = G.predecessors(n)
        succs = G.successors(n)
        for p in preds:
            for s in succs:
                e0, e1 = edge_id_lookup_tbl[(p, n)], edge_id_lookup_tbl[(n, s)]
                edge_strength_mat[e0][e1] = \
                    _strength_edges(edge_weight_vec[e0], edge_weight_vec[e1], node_weight_vec[n01])
                # record edge-edge intersection
                if e0 not in edge_node_map:
                    edge_node_map[e0] = {}
                edge_node_map[e0][e1] = n01

    return node_strength_mat, node_edge_map, edge_strength_mat, edge_node_map


def tacsim(G1, G2=None, node_attribute='weight', edge_attribute='weight', max_iter=100, eps=1e-4, tol=1e-6):
    """ Calculate the TACSim measure of two attributed, directed graph.
    """
    if isinstance(G1, nx.MultiDiGraph):
        assert("MultiDiGraph is not supported by TACSim.")

    nsm1, nem1, esm1, enm1 = _graph_elements(G1, node_attribute, edge_attribute)

    if G2 is None:
        nsm2, nem2, esm2, enm2 = nsm1, nem1, esm1, enm1
        G2 = G1
    else:
        nsm2, nem2, esm2, enm2 = _graph_elements(G2, node_attribute, edge_attribute)

    N = len(G1.nodes())
    M = len(G2.nodes())
    nsim_prev = np.zeros((N, M))
    nsim = np.ones((N, M))

    P = len(G1.edges())
    Q = len(G2.edges())
    esim_prev = np.zeros((P, Q))
    esim = np.ones((P, Q))

    for itrc in range(max_iter):
        if _converged(nsim, nsim_prev, esim, esim_prev):
            break

        nsim_prev = copy.deepcopy(nsim)
        esim_prev = copy.deepcopy(esim)

        # Update node similarity, in and out node neighbors
        for i, j in itertools.product(range(N), range(M)):
            u_in = [u for u in range(N) if nsm1[u,i] >= 0]
            v_in = [v for v in range(M) if nsm2[v,j] >= 0]
            for u, v in itertools.product(u_in, v_in):
                u_edge = nem1[u][i]
                v_edge = nem2[v][j]
                nsim[i][j] += 0.5 * _coherence(nsm1[u,i], nsm2[v,j]) * (nsim_prev[u,v] + esim_prev[u_edge][v_edge])

            u_out = [u for u in range(N) if nsm1[i,u] >= 0]
            v_out = [v for v in range(M) if nsm2[j,v] >= 0]
            for u, v in itertools.product(u_out, v_out):
                u_edge = nem1[i][u]
                v_edge = nem2[j][v]
                nsim[i][j] += 0.5 * _coherence(nsm1[i,u], nsm2[j,v]) * (nsim_prev[u,v] + esim_prev[u_edge][v_edge])

        # Update edge similarity, in and out edge neighbors
        for i, j in itertools.product(range(P), range(Q)):
            u_in = [u for u in range(P) if esm1[u,i] >= 0]
            v_in = [v for v in range(Q) if esm2[v,j] >= 0]
            for u, v in itertools.product(u_in, v_in):
                u_node = enm1[u][i]
                v_node = enm2[v][j]
                esim[i][j] += 0.5 * _coherence(esm1[u,i], esm2[v,j]) * (esim_prev[u,v] + nsim_prev[u_node][v_node])

            u_out = [u for u in range(P) if esm1[i,u] >= 0]
            v_out = [v for v in range(Q) if esm2[j,v] >= 0]
            for u, v in itertools.product(u_out, v_out):
                u_node = enm1[i][u]
                v_node = enm2[j][v]
                esim[i][j] += 0.5 * _coherence(esm1[i,u], esm2[j,v]) * (esim_prev[u,v] + nsim_prev[u_node][v_node])

        nsim = normalized(nsim)
        esim = normalized(esim)

    print("Converge after %d iterations (eps=%f)." % (itrc, eps))

    return _mask_lower_values(nsim, tol), _mask_lower_values(esim, tol)


@params(G=nx.DiGraph)
def node_edge_adjacency(G):
    """ Node-edge adjacency matrix: source nodes
    """
    edges = G.edges()
    nodes = G.nodes()
    node_index = {}
    for i in range(0, len(nodes)):
        node_index[nodes[i]] = i

    ne_src_mat = np.zeros([len(nodes), len(edges)])
    ne_dst_mat = np.zeros([len(nodes), len(edges)])

    for i in range(0, len(edges)):
        s, t = edges[i]
        ne_src_mat[node_index[s]][i] = 1
        ne_dst_mat[node_index[t]][i] = 1

    return ne_src_mat, ne_dst_mat


def tacsim_combined(G1, G2=None, node_attribute='weight', edge_attribute='weight', lamb = 0.5, norm=True):
    """ Combined similarity based on original tacsim scores. Refer to paper Mesos.
    """
    # X: node similarity; Y: edge similarity
    X, Y = tacsim(G1, G2, node_attribute, edge_attribute)

    As, At = node_edge_adjacency(G1)
    if G2 is None:
        Bs, Bt = As, At
    else:
        Bs, Bt = node_edge_adjacency(G2)

    Z = Y + lamb * np.dot(np.dot(As.T, X), Bs) + (1-lamb) * np.dot(np.dot(At.T, X), Bt)

    if norm:
        return normalized(Z)
    else:
        return Z


if __name__ == '__main__':
    G1 = nx.DiGraph()
    G1.add_weighted_edges_from([(1,0,8), (0,2,12), (1,2,10), (2,3,15)])
    G1.node[0]['weight'] = 1
    G1.node[1]['weight'] = 1
    G1.node[2]['weight'] = 5
    G1.node[3]['weight'] = 1

    G2 = nx.DiGraph()
    G2.add_weighted_edges_from([(0,1,15), (1,2,10)])
    G2.node[0]['weight'] = 1
    G2.node[1]['weight'] = 3
    G2.node[2]['weight'] = 1

    print(tacsim(G1, G2))
    print(tacsim(G1))

    print(tacsim_combined(G1, G2))
