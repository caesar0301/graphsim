graphsim
--------

Graph similarity algorithms based on NetworkX.

**BSD License**


Install
-------

    $ pip install -U graphsim


Usage
-----

    >>> import graphsim as gs


Supported algorithms
--------------------

* `gs.ascos`: Asymmetric network Structure COntext Similarity, by Hung-Hsuan Chen et al.
* `gs.nsim_bvd04`: node-node similarity matrix, by Blondel et al.
* `gs.hits`: the hub and authority scores for nodes, by Kleinberg.
* `gs.nsim_hs03`: node-node similarity with mismatch penalty, by Heymans et al.
* `gs.simrank`: A Measure of Structural-Context Similarity, by Jeh et al.
* `gs.simrank_bipartite`: SimRank for bipartite graphs, by Jeh et al.
* `gs.tacsim`: Topology-Attributes Coupling Similarity, by Xiaming Chen et al.
* `gs.tacsim_self`: calculate the self-similarity via TACSim.
* `gs.tacsim_combined`: A combined topology-attributes coupling similarity, by Xiaming Chen et al.
* `gs.tacsim_in_C`: an efficient implementation of TACSim in pure C.
* `gs.tacsim_self_in_C`: an efficient implementation of self-similarity via TACSim in C.


Supported utilities
-------------------

* `gs.normalized`: L2 normalization of vectors, matrices or arrays.
* `gs.node_edge_adjacency`: Obtain node-edge adjacency matrices in source and dest directions.


Author
------

Xiaming Chen <chen@xiaming.me>