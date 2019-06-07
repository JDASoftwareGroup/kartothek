===============
Getting started
===============

``kartothek`` offers a metadata definition to handle these datasets efficiently.

Datasets in ``kartothek`` are made up of one or more ``tables``, each with a unique schema.
When working with ``kartothek`` tables as a Python user, we will use :class:`pandas.DataFrame`
as the user-facing type.

We typically expect that the contents of a dataset are
large, often too large to be held in memory by a single machine but for demonstration
purposes, we will use a small DataFrame with a mixed set of types.

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

Defining the storage location
=============================

We want to store this DataFrame now as a dataset. Therefore, we first need
to connect to a storage location. ``kartothek`` can write to any location that
fulfills the `simplekv.KeyValueStore interface`_. We use `storefact`_ in this
example to construct such a store for the local filesystem.

.. ipython:: python

    from functools import partial
    from storefact import get_store_from_url
    from tempfile import TemporaryDirectory

    dataset_dir = TemporaryDirectory()

    store_factory = partial(get_store_from_url, f"hfs://{dataset_dir.name}")

The reason ``store_factory`` is defined as a ``partial`` callable with the store
information as arguments is because,
when using distributed computing backends in ``kartothek``,
the connections of the store cannot be safely transferred between processes,
thus, we pass storage information to workers as a factory function.


Writing dataset to storage
===========================

Now that we have our data and the storage location, we can persist the dataset.
To do so, we will use :func:`kartothek.io.eager.store_dataframes_as_dataset`
to store a ``DataFrame`` we already have in memory.

.. ipython:: python

   from kartothek.io.eager import store_dataframes_as_dataset
   dm = store_dataframes_as_dataset(
      store, "a_unique_dataset_identifier", df, metadata_version=4
   )
   dm

After calling :func:`~kartothek.io.eager.store_dataframes_as_dataset`,
a :class:`kartothek.core.dataset.DatasetMetadata` object is returned.
This class holds information about the structure and schema of the dataset.

For this guide, two attributes that are noteworthy are ``tables`` and ``partitions``:

- Each dataset has one or more ``tables``, where each table represents a particular subset of
  data, this data is stored as a collection of dataframes/files which have the same schema.
- Data is written to storage in batches (for ``eager``, there is only a single batch),
  in this sense a batch is termed a ``partition`` in ``kartothek``. An alternative to
  this approach of partitioning takes place when the user specifies the column on which
  they would like the dataset to be partitioned. In this case, there will exist a partition
  for each different value of that column, for each batch of data.
  Partitions are structurally identical to each other and each partition
  is made up of a collection of datframes/files containing the subset of data of each table
  belonging to that partition.


For each table, ``kartothek`` also tracks the schema of the columns.
Unless specified explicitly on write, it is inferred from the passed data.
On writing additional data to a dataset, we will also check that the schema
of the new data matches the schema of the existing data.

.. admonition:: Passing multiple partitions to a dataset during write/update

    To store multiple dataframes into a dataset (i.e. multiple `partitions`), it is possible
    to pass an iterator of dataframes, the exact format will depend on the I/O backend used.

    If passing an iterator of dataframes, and table names are not specified, ``kartothek``
    assumes these dataframes are different partitions with a single table.

A ``ValueError`` will be thrown if there is a mismatch in the schema. For example,
passing a list of dataframes with differing schemas and without table names to
:func:`~kartothek.io.eager.store_dataframes_as_dataset`.

As we have not explicitly defined the name of the table nor the name
of the created partition, ``kartothek`` has used the default table name
``table`` and generated a UUID for the partition name.

.. ipython:: python

    dm.tables
    dm.partitions


.. admonition:: A more complex example: multiple tables and partitions

    Sometimes it may be useful to write multiple dataframes with different schemas into
    a single dataset. This can be achieved by creating a dataset with multiple tables.

    In this example, we create a dataset with two partitions (represented by
    the dictionary objects inside the list).
    For each partition, there exist two tables: ``core-table`` and ``aux-table``.
    The schemas of the tables are identical across partitions.

    .. ipython:: python

       dfs = [
            {
                "data": {
                    "core-table": pd.DataFrame({"col1": ["x"]}),
                    "aux-table": pd.DataFrame({"f": [1.1]}),
                },
            },
            {
                "data": {
                    "core-table": pd.DataFrame({"col1": ["y"]}),
                    "aux-table": pd.DataFrame({"f": [3.2]}),
                },
            },
       ]

       store_dataframes_as_dataset(store_factory, dataset_uuid="two-tables", dfs=dfs)


Reading dataset from storage
=============================

After we have written the data, we may want to read it back in again. For this we can
use :func:`kartothek.io.eager.read_table`. This method returns the complete
table of the dataset as a pandas DataFrame (since there is only a single table in this
example, it returns the entire dataset).

.. ipython:: python

    from kartothek.io.eager import read_table

    read_table("a_unique_dataset_identifier", store_factory, table="table")


We can also read a dataframe iteratively, using
:func:`kartothek.io.iter.read_dataset_as_dataframes__iterator`. This will return a generator
of dictionaries (one dictionary for each `partition`), where the keys of each dictionary
represent the `tables` of the dataset. For example,

.. ipython:: python

    from kartothek.io.iter import read_dataset_as_dataframes__iterator

    for partition_index, df_dict in enumerate(
            read_dataset_as_dataframes__iterator(dataset_uuid="two-tables", store=store_factory)
        ):
            print(f"Partition #{partition_index}")
            for table_name, table_df in df_dict.items():
                print(f"Table: {table_name}. Data: \n{table_df}")


.. admonition:: Filtering the dataset using predicates

    It is possible to perform reading queries similar to an SQL ``WHERE`` statement using
    the ``predicates`` argument. When this argument is defined, ``kartothek``
    uses the Apache Parquet metadata to load only chunks of the dataset which may contain values
    that fullfill the query. Such a query will be significantly faster if the dataset is
    partitioned or has an index built on the column queried.

    .. ipython:: python

        # Read only values table `aux-table` where `f` < 2.5
        read_table(
            "two-tables", store_factory, table="aux-table", predicates=[[("f", "<", 2.5)]]
        )



Updating an existing dataset
============================

It's possible to update datasets by adding new physical partitions to them, ``kartothek`` provides
update functions that generally have the prefix `update_dataset` in their names.
For example, :func:`kartothek.io.eager.update_dataset_from_dataframes` is the update
function for the ``eager`` backend.

To update data to an existing dataset in the ``eager`` backend, we
create ``another_df`` with the same schema as our intial dataframe,
and call :func:`~kartothek.io.eager.update_dataset_from_dataframes`:

.. ipython:: python

    from kartothek.io.eager import update_dataset_from_dataframes

    another_df = pd.DataFrame(
        {
            "A": 5.,
            "B": pd.Timestamp("20110102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([12] * 4, dtype="int32"),
            "E": pd.Categorical(["prod", "train", "test", "train"]),
            "F": "bar",
        }
    )

    dm = update_dataset_from_dataframes(
        [another_df],
        store=store_factory,
        dataset_uuid="a_unique_dataset_identifier"
    )
    dm.partitions

Looking at ``dm.partitions``, we can see that another partition has
been added.

If we read the data again, we can see that the ``another_df`` has been appended to the
previous contents.

.. ipython:: python

    df_again = read_table("a_unique_dataset_identifier", store_factory, table="table")
    df_again


The way dataset updates work is that new partitions are added to a dataset
as long as they have the same tables as the existing partitions. A `different`
table **cannot** be introduced into an existing dataset with an update.

Below is an example, where one updates an existing dataset with multiple tables:

.. ipython:: python

    another_df2 = pd.DataFrame(
        {
            "G": "bar",
            "H": pd.Categorical(["test", "train", "test", "train"]),
            "I": np.array([6] * 4, dtype="int32"),
            "J": pd.Series(2, index=list(range(4)), dtype="float32"),
            "K": pd.Timestamp("20190604"),
            "L": 2.,
        }
    )
    another_df2

    dm = update_dataset_from_dataframes(
        {
            "data":
            {
                "table1": another_df,
                "table2": another_df2
            }
        },
        store=store_factory,
        dataset_uuid="another_unique_dataset_identifier"
    )
    dm


Trying to update only a subset of tables throws a ``ValueError``:

.. ipython::

   @verbatim
   In [45]: update_dataset_from_dataframes(
      ....:        {
      ....:           "data":
      ....:           {
      ....:              "table2": another_df2
      ....:           }
      ....:        },
      ....:        store=store_factory,
      ....:        dataset_uuid="another_unique_dataset_identifier"
      ....:        )
      ....:
   ---------------------------------------------------------------------------
   ValueError: Input partitions for update have different tables than dataset:
   Input partition tables: {'table2'}
   Tables of existing dataset: ['table1', 'table2']


Partitioning and Indexing
=========================

``kartothek`` is designed primarily for storing large datasets consistently and
accessing them efficiently. To achieve this, it provides two useful functionalities:
partitioning and secondary indices.

Partitioning
------------

As we have already seen, updating a dataset in ``kartothek`` amounts to adding new
partitions, which in the underlying key-value store translates to writing new files
to the storage layer.

From the perspective of efficient access, it would be helpful if accessing a subset
of written data didn't require reading through an entire dataset to be able to identify
and access the required subset. This is where partitioning by table columns helps.

Specifically, ``kartothek`` allows users to (physically) partition their data by the
values of table columns such that all the rows with the same value of the column all get
written to the same partition. To do this, we use the ``partition_on`` keyword argument:

.. ipython:: python

    df = another_df

    dm = store_dataframes_as_dataset(
        store_factory,
        "partitioned_dataset",
        df,
        partition_on = 'E',
    )
    dm

Of interest here is ``dm.partitions``:

.. ipython:: python

    dm.partitions

We can see that partitions have been stored in a way which indicates the
specific value for the column on which partitioning has been performed.

Partitioning can also be performed on multiple columns; in this case, columns should
be specified as a list:

.. ipython:: python

    dm = store_dataframes_as_dataset(
        store_factory,
        "another_partitioned_dataset",
        [df, another_df],
        partition_on = ['E', 'F'],
    )
    dm.partitions

Note that, since 2 dataframes have been provided as input to the function, there are
4 different files created, even though only 2 different combinations of values of E and
F are found, ``E=test/F=foo`` and ``E=train/F=foo`` .
(However, these 4 physical partitions can be read as just the 2 logical partitions by
using the argument ``concat_partitions_on_primary_index=True`` at reading time).

For datasets consisting of multiple tables, partitioning on
columns only works if the column exists in both tables and is of the same data type.

For example,

.. ipython:: python

    df.dtypes
    df3 = pd.DataFrame(
        {
            "B": pd.to_datetime(["20130102","20190101"]),
            "L": [1, 4],
            "Q": [True, False],
        }
    )
    df3.dtypes

    dm = store_dataframes_as_dataset(
        store_factory,
        "multiple_partitioned_tables",
        [{"data": {"table1": df, "table2": df3}}],
        partition_on='B',
    )

    dm.partitions


Because partitions are physical in nature, it is not possible to modify the
partitioning scheme of an existing dataset via an update, instead, the dataset would have to be
re-created.

Secondary Indices
-----------------

The ability to build and maintain secondary indices are an additional ability
provided by ``kartothek``. Secondary indices are `similar` to partitions in the
sense that they allow faster access to subsets of data. The main difference
between them is that while partitioning actually creates separate partitions based
on column values, secondary indices are simply python dictionaries mapping column
values and the partitions that rows with them can be found in.

.. note::

    The examples we've looked at so far have mostly used functions from the ``eager``
    backend. As noted earlier, the ``iter`` backend executes operations on the dataset
    on a per-partition basis and accordingly data inputs are expected to be generators.
    Although using other iterables such as lists also works, doing so is counter
    to the intent of the ``iter`` backend (lists would be appropriate in ``eager``).

Writing a dataset with a secondary index:

.. ipython:: python

    from kartothek.io.iter import store_dataframes_as_dataset__iter

    # "Generate" 5 dataframes
    df_gen = (
        pd.DataFrame(
            {
                "date": pd.Timestamp(f"2020-01-0{i}"),
                "X": np.random.choice(10, 10),
            }
        )
        for i in range(1, 6)
    )

    dm = store_dataframes_as_dataset__iter(
        df_gen,
        store_factory,
        "secondarily_indexed",
        partition_on = "date",
        secondary_indices = "X"
    )
    dm

    dm = dm.load_all_indices(store_factory())
    dm.secondary_indices['X'].index_dct[0]  # Show files where `X == 0`

As can be seen from the example above, both ``partition_on`` and ``secondary_indices``
can be specified together. Multiple ``secondary_indices`` can also be added as a list of
strings.

In general, secondary indices behave like partitions in terms of when and how they can
and cannot be created. However, when using ``partition_on`` the order of the columns
provided is important, whereas it is ignored for ``secondary_indices``.

Garbage collection
==================

When ``kartothek`` is executing an operation, it makes sure to not
commit changes to the dataset until the operation has been succesfully completed. If a
write operation does not succeed for any reason, although there may be new files written
to storage, those files will not used by the dataset as they will not be referenced in
the ``kartothek`` metadata. Thus, when the user reads the dataset, no new data will
appear in the output.

Similarly, when deleting a partition, ``kartothek`` only removes the reference of that file
from the metadata.


These temporary files will remain in storage until a ``kartothek``  garbage collection
function is called on the dataset.
If a dataset is updated on a regular basis, it may be useful to run garbage collection
periodically to decrease unnecessary storage use.

An example of garbage collection is shown below. A file named ``trash.parquet`` is
created in storage but untracked by kartothek. When garbage collection is called, the
file is removed.

.. ipython:: python

    from kartothek.io.eager import garbage_collect_dataset

    store = store_factory()
    # Put corrupt parquet file in storage for dataset "a_unique_dataset_identifier"
    store.put("a_unique_dataset_identifier/table/trash.parquet", b"trash")
    files_before = set(store.keys())

    garbage_collect_dataset(store=store_factory, dataset_uuid="a_unique_dataset_identifier")

    files_before.difference(store.keys())  # Show files removed


.. _simplekv.KeyValueStore interface: https://simplekv.readthedocs.io/en/latest/#simplekv.KeyValueStore
.. _storefact: https://github.com/blue-yonder/storefact
.. _dask: https://docs.dask.org/en/latest/
