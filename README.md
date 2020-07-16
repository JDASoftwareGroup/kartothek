Kartothek
=========

[![Build Status](https://github.com/JDASoftwareGroup/kartothek/workflows/CI/badge.svg)](https://github.com/JDASoftwareGroup/kartothek/actions?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/kartothek/badge/?version=latest)](https://kartothek.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/github/JDASoftwareGroup/kartothek/coverage.svg?branch=master)](https://codecov.io/github/JDASoftwareGroup/kartothek)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/JDASoftwareGroup/kartothek/blob/master/LICENSE.txt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kartothek/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kartothek/badges/downloads.svg)](https://anaconda.org/conda-forge/kartothek)

Kartothek is a Python library to manage (create, read, update, delete) large
amounts of tabular data in a blob store. It stores data as datasets, which
it presents as pandas DataFrames to the user. Datasets are a collection of
files with the same schema that reside in a blob store. Kartothek uses a metadata
definition to handle these datasets efficiently. For distributed access and
manipulation of datasets Kartothek offers a [Dask](https://dask.org) interface.

Storing data distributed over multiple files in a blob store (S3, ABS, GCS,
etc.) allows for a fast, cost-efficient and highly scalable data infrastructure.
A downside of storing data solely in an object store is that the storages
themselves give little to no guarantees beyond the consistency of a single file.
In particular, they cannot guarantee the consistency of your dataset. If we
demand a consistent state of our dataset at all times, we need to track the
state of the dataset. Kartothek frees us from having to do this manually.

The `kartothek.io` module provides building blocks to create and modify these
datasets in data pipelines. Kartothek handles I/O, tracks dataset partitions
and selects subsets of data transparently.

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
