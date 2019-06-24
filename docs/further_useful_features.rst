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

.. _storage_specification:

Storage specification
=====================

``kartothek`` can write to any location that
fulfills the `simplekv.KeyValueStore interface
<https://simplekv.readthedocs.io/en/latest/#simplekv.KeyValueStore>`_  as long as they
support `ExtendedKeyspaceMixin
<https://github.com/mbr/simplekv/search?q=%22class+ExtendedKeyspaceMixin%22&unscoped_q=%22class+ExtendedKeyspaceMixin%22>`_
(this is necessary so that ``/`` can be used in the storage key name).

For more information, take a look out at the `storefact documentation
<https://storefact.readthedocs.io/en/latest/reference/storefact.html>`_.

The reason ``store_factory`` is defined as a ``partial`` callable with the store
information as arguments is because, when using distributed computing backends in
``kartothek``, the connections of the store cannot be safely transferred between
processes and thus we pass storage information to workers as a factory function.

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
of written data didn't require reading through an entire dataset to be able to
identify and access the required subset. This is where *explicitly* partitioning by
table columns helps.

To see explicit partitioning in action, let's set up some data and a storage location
first and store the data there with ``kartothek``:

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


``kartothek`` allows users to explicitly partition their data by the values of table
columns such that, for a given input partition, all the rows with the same value of the
column all get written to the same partition. To do this, we use the
``partition_on`` keyword argument:

.. ipython:: python

    dm = store_dataframes_as_dataset(
        store_factory, "partitioned_dataset", df, partition_on="E"
    )


Of interest here is ``dm.partitions``:

.. ipython:: python

    sorted(dm.partitions.keys())


We can see that partitions have been stored in a way which indicates the
specific value for the column on which partitioning has been performed.

Partitioning can also be performed on multiple columns; in this case, columns
should be specified as a list:

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
F are found, ``E=test/F=foo`` and ``E=train/F=foo`` (However, these 4 physical partitions
can be read as just the 2 logical partitions by using the argument
``concat_partitions_on_primary_index=True`` at reading time).

For datasets consisting of multiple tables, explicit partitioning on columns can only be
performed if the column exists in both tables and is of the same data type: guaranteeing
that their types are the same is part of schema validation in ``kartothek``.

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


As noted above, when data is appended to a dataset, ``kartothek`` guarantees it has
the proper schema and partitioning.

The order of columns provided in ``partition_on`` is important, as the partition
structure would be different if the columns are in a different order.

.. note:: Every partition must have data for every table. An empty dataframe in this
          context is also considered as data.


Secondary Indices
-----------------

The ability to build and maintain `inverted indices <https://en.wikipedia.org/wiki/Inverted_index>`_
are an additional feature provided by ``kartothek``.
In general, an index is a data structure used to improve
the speed of read queries. In the context of ``kartothek`` an index is a data structure
that contains a mapping of every unique value of a given column to references to all the
partitions where this value occurs.

While this index has a one-to-one mapping of column values to partition references,
secondary indices have the advantage of being able to contain one-to-many mappings of
column values to partition references.

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

Contrary to ``partition_on``, the order of columns is ignored for ``secondary_indices``.


Updating existing data
======================

It's possible to update existing data (formally referred to in ``kartothek`` as a dataset)
by adding new physical partitions to them and deleting or replacing old partitions. ``kartothek``
provides update functions that generally have the prefix `update_dataset` in their names.
For example, :func:`~kartothek.io.eager.update_dataset_from_dataframes` is the update
function for the ``eager`` backend.

To see updating in action, let's first set up a storage location first and store some data there
with ``kartothek``. Specifically, we'll reuse the ``df`` dataframe that we'd created earlier:

.. ipython:: python

    dm = store_dataframes_as_dataset(store_factory, "a_unique_dataset_identifier", df)
    sorted(dm.partitions.keys())


Appending Data
--------------

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

To illustrate this point better, let's first create a dataset with two tables:

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


.. admonition:: Partition names

   In the previous example a dictionary was used to pass the desired data to the store function. To label each
   partition, by default ``kartothek`` uses UUIDs to ensure that each partition is named uniquely. This is
   necessary so that the update can properly work using `copy-on-write <https://en.wikipedia.org/wiki/Copy-on-write>`_
   principles.

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


Deleting Data
-------------

Adding data to an existing dataset is not the only functionality achievable within an update
operation, and it can also be used to remove data.
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

.. warning:: We defined ``delete_scope`` over a value of ``E`` which is also the column that
    we partitioned on: ``delete_scope`` *only works on* indexed columns.

    Furthermore it *should only* be used on partitioned columns due to their one-to-one mapping;
    without the guarantee of one-to-one mappings, using ``delete_scope`` could have unwanted
    effects like accidentally removing data with different values.

    Attempting to use ``delete_scope`` *will also* work on datasets not previously partitioned
    on any column(s); however this is **not at all advised** since the effect will simply be to
    remove **all** previous partitions and replace them with the ones in the update.

    If the intention of the user is to delete *all* existing partitions, using :func:`kartothek.io.eager.delete_dataset`
    would be a much better, cleaner and safer way to go about doing so.


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


Replacing Data
--------------

Finally, an update step can be used to perform the two steps above, i.e. deleting and appending
together in one shot. This is done simply by specifying a dataset to be appended while also defining
a ``delete_scope`` over the partition. The following example illustrates how both can be performed
with one update:

.. ipython:: python

    df  # this has 2 rows where column E has value 'test' and another 2 rows where E is 'train'

    dm = store_dataframes_as_dataset(
        store_factory, "replace_partition", df, partition_on="E"
    )
    sorted(dm.partitions.keys())  # two partitions, one each for E=test and E=train

    modified_df = another_df.copy()
    modified_df.E = (
        "train"
    )  # set column E to have value 'train' for all rows in this dataframe
    modified_df

    dm = update_dataset_from_dataframes(
        [
            modified_df
        ],  # specify dataframe which has 'new' data for partition to be replaced
        store=store_factory,
        dataset_uuid="replace_partition",
        partition_on="E",  # don't forget to specify the partitioning column
        delete_scope=[{"E": "train"}],  # specify the partition to be deleted
    )
    sorted(dm.partitions.keys())

    modified_df = read_table("replace_partition", store_factory, table="table")
    modified_df


As can be seen in the example above, the resultant dataframe from :func:`~kartothek.io.eager.read_table` has two rows
corresponding to ``E=test`` from ``df`` and four rows corresponding to ``E=train`` from ``modified_df``
and the net result is that the original partition with the two rows corresponding to ``E=train`` from ``df``
has been completely replaced.

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

.. _storefact: https://github.com/blue-yonder/storefact
