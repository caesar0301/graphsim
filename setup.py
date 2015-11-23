import os
import sys
from setuptools import setup
from graphsim import __version__
from graphsim import __author__
from graphsim import __email__

setup(
    name = "graphsim",
    version = __version__,
    url = 'https://github.com/caesar0301/graphsim',
    author = __author__,
    author_email = __email__,
    description = 'Graph similarity algorithms based on NetworkX.',
    long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
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
