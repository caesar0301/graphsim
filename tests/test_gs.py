import networkx as nx
import graphsim as gs

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

    print(gs.tacsim_in_C(G1, G2))

    print(gs.tacsim_in_C(G1))

    print(gs.tacsim_combined_in_C(G1, G2))
