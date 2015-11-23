import itertools

import numpy as np
import networkx as nx
from typedecorator import params, returns


__all__ = [ 'similarity_coupling' ]


@params(G1=nx.DiGraph, G2=nx.DiGraph, max_iter=int, eps=float)
def similarity_coupling(G1, G2, max_iter=100, eps=1e-4):
    pass