.. _getting_started:

===============
Getting Started
===============


Kartothek manages datasets that consist of files that contain tables. It does so by offering
a metadata definition to handle these datasets efficiently.

Datasets in Kartothek are made up of one or more ``tables``, each with a unique schema.
When working with Kartothek tables as a Python user, we will use :class:`~pandas.DataFrame`
as the user-facing type.

We typically expect that the contents of a dataset are
large, often too large to be held in memory by a single machine but for demonstration
purposes, we will use a small DataFrame with a mixed set of types.

.. ipython:: python

    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": pd.Timestamp("20130102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "prod"]),
            "F": "foo",
        }
    )

    another_df = pd.DataFrame(
        {
            "A": 5.0,
            "B": pd.Timestamp("20110102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([12] * 4, dtype="int32"),
            "E": pd.Categorical(["prod", "train", "test", "train"]),
            "F": "bar",
        }
    )


Defining the storage location
=============================

We want to store this DataFrame now as a dataset. Therefore, we first need
to connect to a storage location.

We define a store factory as a callable which contains the storage information.
We will use `storefact`_ in this example to construct such a store factory
for the local filesystem (``hfs://`` indicates we are using the local filesystem and
what follows is the filepath).

.. ipython:: python

    from functools import partial
    from tempfile import TemporaryDirectory
    from storefact import get_store_from_url

    dataset_dir = TemporaryDirectory()

    store_url = f"hfs://{dataset_dir.name}"

.. admonition:: Storage locations

    `storefact`_ offers support for several stores in Kartothek, these can be created using the
    function :func:`storefact.get_store_from_url` with one of the following prefixes:

    - ``hfs``: Local filesystem
    - ``hazure``: AzureBlockBlobStorage
    - ``hs3``:  BotoStore (Amazon S3)

Interface
---------

Kartothek can write to any location that
fulfills the `simplekv.KeyValueStore interface
<https://simplekv.readthedocs.io/en/latest/#simplekv.KeyValueStore>`_  as long as they
support `ExtendedKeyspaceMixin
<https://github.com/mbr/simplekv/search?q=%22class+ExtendedKeyspaceMixin%22&unscoped_q=%22class+ExtendedKeyspaceMixin%22>`_
(this is necessary so that ``/`` can be used in the storage key name).

For more information, take a look out at the `storefact documentation
<https://storefact.readthedocs.io/en/latest/reference/storefact.html>`_.


Writing data to storage
=======================

Now that we have some data and a location to store it in, we can persist it as a
dataset. To do so, we will use :func:`~kartothek.io.eager.store_dataframes_as_dataset`
to store the ``DataFrame`` ``df`` that we already have in memory.

.. ipython:: python

    from kartothek.io.eager import store_dataframes_as_dataset

    df.dtypes.equals(another_df.dtypes)  # both have the same schema

    dm = store_dataframes_as_dataset(
        store_url, "a_unique_dataset_identifier", [df, another_df]
    )


.. admonition:: Scheduling backends

    The import path of this function already gives us a hint about the general
    structuring of the Kartothek modules. In :mod:`kartothek.io` we have all
    the building blocks to build data pipelines that read and write from/to storages.
    The next module level (e.g. ``eager``) describes the scheduling backend.

    The scheduling backends `currently supported` by Kartothek are:

    - ``eager`` runs all execution immediately and on the local machine.
    - ``iter`` executes operations on the dataset using a generator/iterator interface.
      The standard format to read/store dataframes in ``iter`` is by providing
      a generator of dataframes.
    - ``dask`` is suitable for larger datasets. It can be used to work on datasets in
      parallel or even in a cluster by using ``dask.distributed`` as the backend.
      There are also ``dask.bag`` and ``dask.dataframe`` which support I/O operations
      for the respective `dask`_ collections.


After calling :func:`~kartothek.io.eager.store_dataframes_as_dataset`,
a :class:`~kartothek.core.dataset.DatasetMetadata` object is returned.
This class holds information about the structure and schema of the dataset.

.. ipython:: python

    dm.tables
    sorted(dm.partitions.keys())
    dm.table_meta["table"].remove_metadata()  # Arrow schema


For this guide, two attributes that are noteworthy are ``tables`` and ``partitions``:

- Each dataset has one or more ``tables``, where each table is a logical collection of data,
  bound together by a common schema.
- ``partitions`` are the physical "pieces" of data which together constitute the
  contents of a dataset. Data is written to storage on a per-partition basis.
  See the section on partitioning for further details: :ref:`partitioning_section`.

The attribute ``table_meta`` can be accessed to see the underlying schema of the dataset.
See :ref:`type_system` for more information.

To store multiple dataframes into a dataset, it is possible to pass a collection of
dataframes; the exact format will depend on the I/O backend used.

Additionally, Kartothek supports several data input formats,
it does not need to always be a plain ``pd.DataFrame``.
See :func:`~kartothek.io_components.metapartition.parse_input_to_metapartition` for
further details.

If table names are not specified when passing an iterator of dataframes,
Kartothek assumes these dataframes are different chunks of the same table
and expects their schemas to be identical. A ``ValueError`` will be thrown otherwise.
For example,

.. ipython:: python

    df2 = pd.DataFrame(
        {
            "G": "foo",
            "H": pd.Categorical(["test", "train", "test", "train"]),
            "I": np.array([9] * 4, dtype="int32"),
            "J": pd.Series(3, index=list(range(4)), dtype="float32"),
            "K": pd.Timestamp("20190604"),
            "L": 2.0,
        }
    )

    df.dtypes.equals(df2.dtypes)  # schemas are different!


.. ipython::

    @verbatim
    In [24]: store_dataframes_as_dataset(
       ....:     store_url,
       ....:     "will_not_work",
       ....:     [df, df2],
       ....: )
       ....:
    ---------------------------------------------------------------------------
    ValueError: Schema violation
    Origin schema: {table/9e7d9217c82b4fda9c4e720dc987c60d}
    Origin reference: {table/80feb4d84ac34a9c9d08ba48c8170647}


.. note:: Read these sections for more details: :ref:`type_system`, :ref:`dataset_spec`,
          :ref:`input_output`.


When we do not explicitly define the name of the table and partition, Kartothek uses the
default table name ``table`` and generates a UUID for the partition name.

.. admonition:: A more complex example: multiple named tables

    Sometimes it may be useful to write multiple dataframes with different schemas into
    a single dataset. This can be achieved by creating a dataset with multiple tables.

    In this example, we create a dataset with two tables: ``core-table`` and ``aux-table``.
    The schemas of the tables are identical across partitions (each dictionary in the
    ``dfs`` list argument represents a partition).

    .. ipython:: python

        dfs = [
            {
                "data": {
                    "core-table": pd.DataFrame({"id": [22, 23], "f": [1.1, 2.4]}),
                    "aux-table": pd.DataFrame({"id": [22], "col1": ["x"]}),
                }
            },
            {
                "data": {
                    "core-table": pd.DataFrame({"id": [29, 31], "f": [3.2, 0.6]}),
                    "aux-table": pd.DataFrame({"id": [31], "col1": ["y"]}),
                }
            },
        ]

        dm = store_dataframes_as_dataset(store_url, dataset_uuid="two-tables", dfs=dfs)
        dm.tables


Reading data from storage
=========================

After we have written the data, we may want to read it back in again. For this we can
use :func:`~kartothek.io.eager.read_table`. This method returns the complete
table of the dataset as a pandas DataFrame.

.. ipython:: python

    from kartothek.io.eager import read_table

    read_table("a_unique_dataset_identifier", store_url, table="table")


We can also read a dataframe iteratively, using
:func:`~kartothek.io.iter.read_dataset_as_dataframes__iterator`. This will return a generator
of dictionaries (one dictionary for each `partition`), where the keys of each dictionary
represent the `tables` of the dataset. For example,

.. ipython:: python

    from kartothek.io.iter import read_dataset_as_dataframes__iterator

    for partition_index, df_dict in enumerate(
        read_dataset_as_dataframes__iterator(dataset_uuid="two-tables", store=store_url)
    ):
        print(f"Partition #{partition_index}")
        for table_name, table_df in df_dict.items():
            print(f"Table: {table_name}. Data: \n{table_df}")

Respectively, the ``dask.delayed`` back-end provides the function
:func:`~kartothek.io.dask.delayed.read_dataset_as_delayed`, which has a very similar
interface to the :func:`~kartothek.io.iter.read_dataset_as_dataframes__iterator`
function but returns a collection of ``dask.delayed`` objects.


.. admonition:: Filtering using predicates

    It is possible to filter data during reads using simple predicates by using
    the ``predicates`` argument. Technically speaking, Kartothek supports predicates
    in `disjunctive normal form <https://en.wikipedia.org/wiki/Disjunctive_normal_form>`_.

    When this argument is defined, Kartothek uses the Apache Parquet metadata
    as well as indices and partition information to speed up queries when possible.
    How this works is a complex topic, see :ref:`efficient_querying`.

    .. ipython:: python

        # Read only values table `core-table` where `f` < 2.5
        read_table("two-tables", store_url, table="core-table", predicates=[[("f", "<", 2.5)]])


For a deeper dive into Kartothek you can take a look at
:ref:`further_useful_features`. Also look at :ref:`efficient_querying`.

.. _storefact: https://github.com/blue-yonder/storefact
.. _dask: https://docs.dask.org/en/latest/
