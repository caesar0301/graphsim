graphsim
--------

Graph similarity algorithms based on NetworkX.

**BSD License**


Install
-------

First, install building tool:

    $ yum install -y scons

On Mac OS:

    $ brew install scons

Then install graphsim via PyPI:

    $ pip install -U graphsim


**NOTE**: `libtacsim` was tested on Ubuntu 12.04, CentOS 6.5 and Mac OS 10.11.2.


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
* `gs.tacsim_combined`: A combined topology-attributes coupling similarity, by Xiaming Chen et al.
* `gs.tacsim_in_C`: an efficient implementation of TACSim in pure C.
* `gs.tacsim_combined_in_C`: an efficient implementation of combined TACSim in pure C.


Supported utilities
-------------------

* `gs.normalized`: L2 normalization of vectors, matrices or arrays.
* `gs.node_edge_adjacency`: Obtain node-edge adjacency matrices in source and dest directions.

Citation
----------

```latex
@article{Chen2017_mesos,
title = "Discovering and Modeling Meta-Structures in Human Behavior from City-Scale Cellular Data",
journal = "Pervasive and Mobile Computing ",
year = "2017",
author = "Xiaming Chen and Haiyang Wang and Siwei Qiang and Yongkun Wang and Yaohui Jin",
```

Author
------

Xiaming Chen <chenxm35@gmail.com>
