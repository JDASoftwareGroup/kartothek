kartothek
=========

[![Build Status](https://travis-ci.org/JDASoftwareGroup/kartothek.svg?branch=master)](https://travis-ci.org/JDASoftwareGroup/kartothek)
[![Documentation Status](https://readthedocs.org/projects/kartothek/badge/?version=latest)](https://kartothek.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/github/JDASoftwareGroup/kartothek/coverage.svg?branch=master)](https://codecov.io/github/JDASoftwareGroup/kartothek)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/JDASoftwareGroup/kartothek/blob/master/LICENSE.txt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kartothek/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kartothek/badges/downloads.svg)](https://anaconda.org/conda-forge/kartothek)

Datasets are a collection of files with the same schema that reside in
a storage. `kartothek` offers a metadata definition to handle these datasets
efficiently. In addition, the `kartothek.io` module provides building
blocks to create and modify these datasets. Handling of I/O, tracking of
dataset partitions and selecting subsets of data are handled transparently.

Installation
---------------------------
Installers for the latest released version are availabe at the [Python
package index](https://pypi.org/project/kartothek) and on conda.

```sh
# Install with pip
pip install kartothek
```

```sh
# Install with conda
conda install -c conda-forge kartothek
```

What is a (real) Kartothek?
---------------------------

A Kartothek (or more modern: Zettelkasten/Katalogkasten) is a tool to organize
(high-level) information extracted from a source of information.
