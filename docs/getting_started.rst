Getting started
===============

``kartothek`` manages datasets that consist of files that contain tables.
When working with these tables as a Python user, we will use pandas DataFrames
as the user-facing type. We typically expect that the dataset contents are
large, often too large to be held in a single machine but for demonstration
purposes, we use a small DataFrame with a mixed set of types.

.. ipython:: python

   import numpy as np
   import pandas as pd

   df = pd.DataFrame(
       {
           "A": 1.,
           "B": pd.Timestamp("20130102"),
           "C": pd.Series(1, index=list(range(4)), dtype="float32"),
           "D": np.array([3] * 4, dtype="int32"),
           "E": pd.Categorical(["test", "train", "test", "train"]),
           "F": "foo",
       }
   )
   df

We want to store this DataFrame now as a dataset. Therefore, we first need
to connect to a storage location. ``kartothek`` can write to any location that
fulfills the `simplekv.KeyValueStore interface`_. We use `storefact`_ in this
example to construct such a store for the local filesystem.

.. ipython:: python

   from storefact import get_store_from_url
   from tempfile import TemporaryDirectory

   dataset_dir = TemporaryDirectory()
   store = get_store_from_url(f"hfs://{dataset_dir.name}")

Now that we have our data and the storage location, we can persist it to a dataset.
For that we use in this guide :func:`kartothek.io.eager.store_dataframes_as_dataset`
to store a ``DataFrame`` we already have in memory in the local task.

The import path of this function already gives us a hint about the general
structuring of the ``kartothek`` modules. In :mod:`kartothek.io` we have all
the building blocks to build data pipelines that read and write from/to storages.
Other top-level modules for example handle the serialization of DataFrames to
``bytes``.

The next module level ``eager`` describes the scheduling backend.
``eager`` runs all execution immediately and on the local machine.
There is also ``iter`` that supports reading the dataset on
a per-partition base. For larger dataset, ``dask`` can be used
to work on datasets in parallel or even in a cluster by using
``distributed`` as the backend for ``dask``.

.. ipython:: python
   :okwarning:

   from kartothek.io.eager import store_dataframes_as_dataset
   dm = store_dataframes_as_dataset(
      store, "a_unique_dataset_identifier", df, metadata_version=4
   )
   dm

After calling :func:`~kartothek.io.eager.store_dataframes_as_dataset`,
a :class:`kartothek.core.dataset.DatasetMetadata` object is returned. This is the main
class holding all information about the parts and schema of the dataset.

The most interesting ones for this guide are ``tables`` and ``partitions``.
Each dataset can have multiple tables, each table is a collection of files
that all have the same schema. These files are called ``partitions`` in
``kartothek``.

As we neither have explicitly defined the name of the table nor the name
of the created partition, ``kartothek`` has used the default table name
``table`` and used a generated UUID for the partition name.

.. ipython:: python

   dm.tables
   dm.partitions

For each table, ``kartothek`` also tracks the schema of the columns.
When not specified explicitly on write, it is inferred from the passed data.
On writing additional data to a dataset, we will also check that the schema
of the new data matches the schema of the existing data. If it doesn't, we will
raise an exception.

The schema is a ``pyarrow.Schema`` object which persists the native Arrow types
for each column. Additionally, the schema also stores infomartion about the Pandas
types and indices. This information is solely of informative nature and is not
used by ``kartothek`` itself.

.. ipython:: python

   dm.table_meta

After we have written the data, we want to read it back in again. For this we
use :func:`kartothek.io.eager.read_table`. This method
returns the whole dataset as a pandas DataFrame and the metadata of the
dataset. The metadata of a dataset is a dict where one can store arbitrary
information about the dataset. As this metadata is always loaded on accessing
the dataset, this should be kept small.


.. ipython:: python
   :okwarning:

   from kartothek.io.eager import read_table

   df = read_table("a_unique_dataset_identifier", store, table="table")
   df

To understand the basics of the dataset, we can look at the files that were
written using the store method. The main file of a dataset is
``<dataset_uuid>.by-dataset-metadata.json``. Here we track all partitions that
exist inside a datasets as well as the tables and additional metadata. This
file is loaded on any operation of the dataset.

The magic ``<dataset_uuid>/<table>/_common_metadata`` file is an Apache Parquet
file that contains no data. It is simply used to persist the schema of a single
table.

Finally ``<dataset_uuid>/<table>/<partition_label>.parquet`` is the file that
contains the data for this partition in the specific table. By default
``kartothek`` serializes data to Apache Parquet files but also supports other
file formats like CSV.

.. ipython:: python

   list(store.keys())

.. _simplekv.KeyValueStore interface: https://simplekv.readthedocs.io/en/latest/#simplekv.KeyValueStore
.. _storefact: https://github.com/blue-yonder/storefact
