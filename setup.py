import os
from setuptools import setup, find_packages

# build libtacsim automatically
rootdir = os.path.dirname(os.path.realpath('__file__'))
moddir = os.path.join(rootdir, 'libtacsim')
os.system('cd %s; sudo scons install; cd -' % moddir)

__version__ = '0.2.7'
__author__ = 'Xiaming Chen'
__email__ = 'chenxm35@gmail.com'


setup(
    name = "graphsim",
    version = __version__,
    url = 'https://github.com/caesar0301/graphsim',
    author = __author__,
    author_email = __email__,
    description = 'Graph similarity algorithms based on NetworkX.',
    long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    license = "BSC License",
    packages = find_packages(),
    keywords = ['graph', 'graph similarity', 'graph matching'],
    install_requires=[
        'networkx',
        'numpy',
        'typedecorator'
    ],
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
