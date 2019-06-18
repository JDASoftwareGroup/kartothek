.. _further_useful_features:

=================================
Further useful kartothek features
=================================


This document introduces additional features of ``kartothek`` that allow users to
partition and index their data during writing, update already written data and
keep their data stores 'clean' with garbage collection.

The intent of this document is to expose users to some additional features of ``kartothek``
in a bid to encourage deeper exploration of ``kartothek`` beyond 'getting started' and help
better enable developers to start thinking about how it can (or cannot) be useful in their
particular applications.


.. _partitioning_section:

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

To see partitioning in action, lets set up some data and a storage location first and store
the data there with ``kartothek``:

.. ipython:: python

    import numpy as np
    import pandas as pd
    from functools import partial
    from tempfile import TemporaryDirectory
    from storefact import get_store_from_url

    from kartothek.io.eager import store_dataframes_as_dataset

    dataset_dir = TemporaryDirectory()

    store_factory = partial(get_store_from_url, f"hfs://{dataset_dir.name}")

    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": pd.Timestamp("20130102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )
    df

``kartothek`` allows users to (physically) partition their data by the
values of table columns such that all the rows with the same value of the column all get
written to the same partition. To do this, we use the ``partition_on`` keyword argument:

.. ipython:: python

    dm = store_dataframes_as_dataset(
        store_factory, "partitioned_dataset", df, partition_on="E"
    )
    dm


Of interest here is ``dm.partitions``:

.. ipython:: python

    sorted(dm.partitions.keys())


We can see that partitions have been stored in a way which indicates the
specific value for the column on which partitioning has been performed.

Partitioning can also be performed on multiple columns; in this case, columns should
be specified as a list:

.. ipython:: python

    duplicate_df = df.copy()
    duplicate_df.F = "bar"

    dm = store_dataframes_as_dataset(
        store_factory,
        "another_partitioned_dataset",
        [df, duplicate_df],
        partition_on=["E", "F"],
    )
    sorted(dm.partitions.keys())


Note that, since 2 dataframes have been provided as input to the function, there are
4 different files created, even though only 2 different combinations of values of E and
F are found, ``E=test/F=foo`` and ``E=train/F=foo`` .
(However, these 4 physical partitions can be read as just the 2 logical partitions by
using the argument ``concat_partitions_on_primary_index=True`` at reading time).

For datasets consisting of multiple tables, partitioning on
columns only works if the column exists in both tables and is of the same data type.

For example:

.. ipython:: python

    df.dtypes
    different_df = pd.DataFrame(
        {"B": pd.to_datetime(["20130102", "20190101"]), "L": [1, 4], "Q": [True, False]}
    )
    different_df.dtypes

    dm = store_dataframes_as_dataset(
        store_factory,
        "multiple_partitioned_tables",
        [{"data": {"table1": df, "table2": different_df}}],
        partition_on="B",
    )

    sorted(dm.partitions.keys())


Because partitions are physical in nature, it is not possible to modify the
partitioning scheme of an existing dataset via an update, instead, the dataset
would have to be re-created.

.. note:: Under the hood, partitions are structurally identical to each other and each partition
    is made up of a collection of files containing the subset of data of each table
    belonging to that partition.

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
        pd.DataFrame({"date": pd.Timestamp(f"2020-01-0{i}"), "X": np.random.choice(10, 10)})
        for i in range(1, 6)
    )

    dm = store_dataframes_as_dataset__iter(
        df_gen,
        store_factory,
        "secondarily_indexed",
        partition_on="date",
        secondary_indices="X",
    )
    dm

    dm = dm.load_all_indices(store_factory())
    dm.indices["X"].eval_operator("==", 0)  # Show files where `X == 0`


As can be seen from the example above, both ``partition_on`` and ``secondary_indices``
can be specified together. Multiple ``secondary_indices`` can also be added as a list of
strings.

In general, secondary indices behave like partitions in terms of when and how they can
and cannot be created. However, when using ``partition_on`` the order of the columns
provided is important, whereas it is ignored for ``secondary_indices``.


Updating an existing dataset
============================

It's possible to update datasets by adding new physical partitions to them, ``kartothek``
provides update functions that generally have the prefix `update_dataset` in their names.
For example, :func:`~kartothek.io.eager.update_dataset_from_dataframes` is the update
function for the ``eager`` backend.

To see updating in action, lets set up some data and a storage location first and store
the data there with ``kartothek``:

.. ipython:: python

    dm = store_dataframes_as_dataset(store_factory, "a_unique_dataset_identifier", df)
    sorted(dm.partitions.keys())


Now, we create ``another_df`` with the same schema as our intial dataframe
``df`` and update it using the ``eager`` backend by calling :func:`~kartothek.io.eager.update_dataset_from_dataframes`:

.. ipython:: python

    from kartothek.io.eager import update_dataset_from_dataframes

    another_df = pd.DataFrame(
        {
            "A": 5.0,
            "B": pd.Timestamp("20110102"),
            "C": pd.Series(2, index=list(range(4)), dtype="float32"),
            "D": np.array([6] * 4, dtype="int32"),
            "E": pd.Categorical(["prod", "dev", "prod", "dev"]),
            "F": "bar",
        }
    )

    dm = update_dataset_from_dataframes(
        [another_df], store=store_factory, dataset_uuid="a_unique_dataset_identifier"
    )
    sorted(dm.partitions.keys())


Looking at ``dm.partitions``, we can see that another partition has
been added.

If we read the data again, we can see that the ``another_df`` has been appended to the
previous contents.

.. ipython:: python

    from kartothek.io.eager import read_table

    updated_df = read_table("a_unique_dataset_identifier", store_factory, table="table")
    updated_df


The way dataset updates work is that new partitions are added to a dataset
as long as they have the same tables as the existing partitions. A `different`
table **cannot** be introduced into an existing dataset with an update.

To illustrate this point better, lets first create a dataset with two tables:

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
    df2

    dm = store_dataframes_as_dataset(
        store_factory,
        "another_unique_dataset_identifier",
        dfs=[{"data": {"table1": df, "table2": df2}}],
    )
    dm.tables
    sorted(dm.partitions.keys())


Below is an example where we update the existing dataset ``another_unique_dataset_identifier``
with new data for ``table1`` and ``table2``:

.. ipython:: python

    another_df2 = pd.DataFrame(
        {
            "G": "bar",
            "H": pd.Categorical(["prod", "dev", "prod", "dev"]),
            "I": np.array([12] * 4, dtype="int32"),
            "J": pd.Series(4, index=list(range(4)), dtype="float32"),
            "K": pd.Timestamp("20190614"),
            "L": 10.0,
        }
    )
    another_df2

    dm = update_dataset_from_dataframes(
        {"data": {"table1": another_df, "table2": another_df2}},
        store=store_factory,
        dataset_uuid="another_unique_dataset_identifier",
    )
    dm.tables
    sorted(dm.partitions.keys())


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


Adding data to an existing dataset is not the only functionality achievable within an update
operation, such an operation can also be used to remove or overwrite data.
To do so we use the ``delete_scope`` keyword argument as shown in the example below:

.. ipython:: python

    dm = update_dataset_from_dataframes(
        None,
        store=store_factory,
        dataset_uuid="partitioned_dataset",
        partition_on="E",
        delete_scope=[{"E": "train"}],
    )
    sorted(dm.partitions.keys())


As we can see, we specified using a dictionary that data where the column ``E`` has the
value ``train`` should be removed. Looking at the partitions after the update, we see that
the partition ``E=train`` has been removed.

.. note:: We defined ``delete_scope`` over a value of ``E``. ``E`` also happens to be a
    column that we partitioned by. This is because using ``delete_scope`` uses the same
    underlying logic as the predicate-based filtering mentioned in :ref:`getting_started`.

    Attempting to use ``delete_scope`` will *also* work on datasets not previously
    partitioned on any column(s); in this case however the effect will simply be to remove
    **all** previous partitions and replace them with the ones in the update.

When  using ``delete_scope``, multiple values for the same column cannot be defined as a
list but have to be specified instead as individual dictionaries, i.e.
``[{"E": ["test", "train"]}]`` will not work but ``[{"E": "test"}, {"E": "train"}]`` will.

.. ipython:: python

    dm = update_dataset_from_dataframes(
        None,
        store=store_factory,
        dataset_uuid="another_partitioned_dataset",
        partition_on=["E", "F"],
        delete_scope=[{"E": "train", "F": "foo"}, {"E": "test", "F": "bar"}],
    )

    sorted(dm.partitions.keys())  # `E=train/F=foo` and `E=test/F=bar` are deleted





Garbage collection
==================

When ``kartothek`` is executing an operation, it makes sure to not
commit changes to the dataset until the operation has been succesfully completed. If a
write operation does not succeed for any reason, although there may be new files written
to storage, those files will not be used by the dataset as they will not be referenced in
the ``kartothek`` metadata. Thus, when the user reads the dataset, no new data will
appear in the output.

Similarly, when deleting a partition, ``kartothek`` only removes the reference of that file
from the metadata.


These temporary files will remain in storage until a ``kartothek``  garbage collection
function is called on the dataset.
If a dataset is updated on a regular basis, it may be useful to run garbage collection
periodically to decrease unnecessary storage use.

An example of garbage collection is shown below. Suppose a file named
``E=train/F=x/d513c388.parquet`` might have been referenced before a deletion or
compaction operation. This file remains in storage but is untracked by kartothek.
When garbage collection is called, the file is removed.

.. ipython:: python

    from kartothek.io.eager import garbage_collect_dataset

    store = store_factory()
    # Put corrupt parquet file in storage for dataset "a_unique_dataset_identifier"
    store.put("a_unique_dataset_identifier/table/E=train/F=x/d513c388.parquet", b"trash")
    files_before = set(store.keys())

    garbage_collect_dataset(store=store_factory, dataset_uuid="a_unique_dataset_identifier")

    files_before.difference(store.keys())  # Show files removed
