=========================================
kartothek - unified metadata for datasets
=========================================

:Release: |release|
:Date: |today|

Datasets are a collection of files with the same schema that reside in
a storage. ``kartothek`` offers a metadata definition to handle these datasets
efficiently. In addition, the :mod:`kartothek.io` module provides building
blocks to create and modify these datasets. Handling of I/O, tracking of
dataset partitions and selecting subsets of data are handled transparently.

To get started, have a look at our :doc:`getting_started` guide, head to the
description of the :doc:`spec/index` or read more about the :doc:`io/index`
module and learn about data pipelines in kartothek.

What is a (real) Kartothek?
---------------------------

A Kartothek (or more modern: Zettelkasten/Katalogkasten) is a tool to organize
(high-level) information extracted from a source of information.

Contents
========

.. toctree::
   :maxdepth: 2

   Getting started <getting_started>
   Further useful kartothek features <further_useful_features>
   Specification <spec/index>
   Partition Indices <spec/partition_indices>
   Type System <spec/type_system>
   In- / Ouptut <io/index>
   Extending <io/extending>
   DataFrame Serialization <spec/serialization>
   Predicate pushdown <spec/predicate_pushdown>
   Module Reference <_rst/modules>
   Versioning <versioning>
   Changelog <changes>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
