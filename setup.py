import os
import sys
from setuptools import setup
from treelib import __version__

setup(
    name = "graphsim",
    version = __version__,
    url = 'https://github.com/caesar0301/graphsim',
    author = 'Xiaming Chen',
    author_email = 'chenxm35@gmail.com',
    description = 'Graph similarity algorithms based on NetworkX.',
    long_description='''
Graph similarity algorithms based on NetworkX.

**BSD License**

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
* `gs.tacsim`: Topology-Attributes Coupling Simmilarity, by Xiaming Chen et al.
''',
    license = "BSC License",
    packages = ['graphsim'],
    keywords = ['graph', 'graph similarity', 'graph matching'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
            'Intended Audience :: Developers',
            'License :: Freely Distributable',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Topic :: Software Development :: Libraries :: Python Modules',
   ],
)
