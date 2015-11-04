"""
Topology-Attributes Coupling Simmilarity (TACSim) measure.
"""
import itertools, copy

import numpy as np
import networkx as nx
from typedecorator import params, returns


__all__ = [ 'tacsim' ]


def _strength_nodes(nw1, nw2, ew):
    return 1.0 * nw1 * nw2 / np.power(ew, 2)


def _strength_edges(ew1, ew2, nw):
    return 1.0 * np.power(nw, 2) / (ew1 * ew2)


def _coherence(s1, s2):
    return 2.0 * np.sqrt(s1 * s2) / (s1 + s2)


def _converged(nsim, nsim_prev, esim, esim_prev, eps=1e-4):
    if np.allclose(nsim, nsim_prev, atol=eps) \
        and np.allclose(esim, esim_prev, atol=eps):
        return True
    return False


def _normalized(a, axis=0, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / l2


def _graph_elements(G, node_attribute='weight', edge_attribute='weight'):
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
        node_weight_vec[i] = G.node[n][node_attribute]
    node_weight_vec = _normalized(node_weight_vec)

    for i, e in enumerate(edges):
        edge_id_lookup_tbl[e] = i
        edge_weight_vec[i] = G.edge[e[0]][e[1]][edge_attribute]
    edge_weight_vec = _normalized(edge_weight_vec)

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


@params(G1=nx.DiGraph, G2=nx.DiGraph, node_attribute=str, edge_attribute=str, max_iter=int, eps=float)
def tacsim(G1, G2, node_attribute='weight', edge_attribute='weight', max_iter=100, eps=1e-4):
    """ Calculate the TACSim measure of two attributed, directed graph.
    """
    if isinstance(G1, nx.MultiDiGraph) or isinstance(G2, nx.MultiDiGraph):
        assert("MultiDiGraph is not supported by TACSim.")

    nsm1, nem1, esm1, enm1 = _graph_elements(G1, node_attribute, edge_attribute)
    nsm2, nem2, esm2, enm2 = _graph_elements(G2, node_attribute, edge_attribute)

    A = nx.adjacency_matrix(G1).todense()
    B = nx.adjacency_matrix(G2).todense()

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

        # Update node similarity
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

        # Update edge similarity
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

        nsim = _normalized(nsim)
        esim = _normalized(esim)

    print("Converge after %d iterations (eps=%f)." % (itrc, eps))

    return nsim, esim


if __name__ == '__main__':
    G1 = nx.DiGraph()
    G1.add_weighted_edges_from([('a','b',10), ('a','c',10), ('b','c',10), ('c','d',20)])
    G1.node['a']['weight'] = 3
    G1.node['b']['weight'] = 1
    G1.node['c']['weight'] = 5
    G1.node['d']['weight'] = 1

    G2 = nx.DiGraph()
    G2.add_weighted_edges_from([('a','b',20), ('b','c',10)])
    G2.node['a']['weight'] = 1
    G2.node['b']['weight'] = 3
    G2.node['c']['weight'] = 1

    nsim, esim = tacsim(G1, G2)
    print nsim
    print esim
