import os
from setuptools import setup, find_packages

__version__ = '0.2.12'
__author__ = 'Xiaming Chen'
__email__ = 'chenxm35@gmail.com'

# build libtacsim automatically
rootdir = os.path.dirname(os.path.realpath('__file__'))
moddir = os.path.join(rootdir, 'libtacsim')
res = os.system('cd %s && scons install && cd -' % moddir)
if res > 0:
    raise RuntimeError('Failed to build libtacsim.')

setup(
    name="graphsim",
    version=__version__,
    url='https://github.com/caesar0301/graphsim',
    author=__author__,
    author_email=__email__,
    description='Graph similarity algorithms based on NetworkX.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    license="BSC License",
    packages=find_packages(),
    keywords=['graph', 'graph similarity', 'graph matching'],
    install_requires=[
        'networkx==1.11',
        'numpy>=1.13',
        'typedecorator>=0.0.4'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
	'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
