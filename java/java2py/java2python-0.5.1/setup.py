#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" setup.py -> java2python setup script.

Simple but effective tool to translate Java source code into Python.

This package provides tools to imperfectly translate Java source code to
Python source code.

This version requires Python 2.7.
"""

from distutils.core import setup
from os import path, listdir


classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Developers
License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)
Natural Language :: English
Operating System :: OS Independent
Operating System :: POSIX
Programming Language :: Python
Programming Language :: Java
Topic :: Software Development
Topic :: Software Development :: Code Generators
"""


description = __doc__.split('\n')[2]
long_description = '\n'.join(__doc__.split('\n')[4:])


def doc_files():
    return [path.join('doc', name) for name in listdir('doc')]


setup(
    name='java2python',
    version='0.5.1',

    description=description,
    long_description=long_description,

    author='Troy Melhase',
    author_email='troy@troy.io',

    url='https://github.com/natural/java2python/',
    download_url='https://github.com/downloads/natural/java2python/java2python-0.5.1.tar.gz',

    keywords=['java', 'java2python', 'compiler'],
    classifiers=filter(None, classifiers.split('\n')),

    packages=[
        'java2python',
        'java2python.compiler',
        'java2python.config',
        'java2python.lang',
        'java2python.lib',
        'java2python.mod',
        ],

    package_data={
        'java2python' : [
            'license.txt',
            'readme.md',
            ],
        'java2python.lang': [
            '*.g',
            '*.tokens',
            ]
        },

    scripts=[
        'bin/j2py',
        ],

    data_files=[
        ('doc', doc_files()),
        ],

    install_requires=['antlr_python_runtime==3.1.3'],

    )
