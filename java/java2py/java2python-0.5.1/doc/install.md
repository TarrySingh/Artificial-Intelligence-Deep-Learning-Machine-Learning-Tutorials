## Installation

### New School

Kids these days have it easy:

    # pip install java2python

### Old School

#### Install ANTLR Runtime

We need the ANTLR Python runtime before we can install java2python:

    # wget http://www.antlr.org/download/antlr-3.1.3.tar.gz
    # tar xfz antlr-3.1.3.tar.gz
    # cd antlr-3.1.3/runtime/Python/
    # python setup.py install

#### Install java2python

Now the goodness:

    # wget https://github.com/downloads/natural/java2python/java2python-0.5.1.tar.gz
    # tar xfz java2python-0.5.1.tar.gz
    # cd java2python
    # python setup.py install

### Development Version

The latest source can be installed directly from github:

    # pip install --upgrade https://github.com/natural/java2python/zipball/master

You'll want to clone or fork the repo to work on the project, however.


### Dependencies

The runtime dependency for java2python is the [Python runtime][] for [ANTLR][].
The exact version number is very important: java2python requires
[version 3.1.3 of the Python runtime][].

The development dependencies (what you need if you're coding java2python) are
[ANTLR][], also version 3.1.3, GNU make, and a JVM.


[version 3.1.3 of the Python runtime]: http://www.antlr.org/download/antlr-3.1.3.tar.gz
[Python runtime]: http://www.antlr.org/wiki/display/ANTLR3/Python+runtime
[ANTLR]: http://www.antlr.org
