
.. _partitioning_section:

Partitioning
============

As we have already seen, writing data in Kartothek amounts to writing
partitions, which in the underlying key-value store translates to writing files
to the storage layer in a structured manner.

From the perspective of efficient access, it would be helpful if accessing a subset
of written data didn't require reading through an entire dataset to be able to
identify and access the required subset. This is where *explicitly* partitioning by
table columns helps.

Kartothek is designed primarily for storing large datasets consistently. One way
to do this is to structure the data well, this can be done by
explicitly partitioning the dataset by select columns.

One benefit of doing so is that it allows for selective operations on data,
which makes reading as well as mutating (replacing or deleting) subsets of data much
more efficient as only a select amount of files need to be read.

To see explicit partitioning in action, let's set up some data and a storage location
first and store the data there with Kartothek:

.. ipython:: python

    import numpy as np
    import pandas as pd
    from functools import partial
    from tempfile import TemporaryDirectory
    from storefact import get_store_from_url

    from kartothek.io.eager import store_dataframes_as_dataset

    dataset_dir = TemporaryDirectory()

    store_url = f"hfs://{dataset_dir.name}"

    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": [
                pd.Timestamp("20130102"),
                pd.Timestamp("20130102"),
                pd.Timestamp("20130103"),
                pd.Timestamp("20130103"),
            ],
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )
    df


Kartothek allows users to explicitly partition their data by the values of table
columns such that, for a given input partition, all the rows with the same value of the
column all get written to the same partition. To do this, we use the
``partition_on`` keyword argument:

.. ipython:: python

    dm = store_dataframes_as_dataset(
        store_url, "partitioned_dataset", [df], partition_on="B"
    )

Partitioning based on a date column ussually makes sense for timeseries data.

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
        store_url,
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
that their types are the same is part of schema validation in Kartothek.

For example:

.. ipython:: python
    :okwarning:

    df.dtypes
    different_df = pd.DataFrame(
        {"B": pd.to_datetime(["20130102", "20190101"]), "L": [1, 4], "Q": [True, False]}
    )
    different_df.dtypes

    dm = store_dataframes_as_dataset(
        store_url,
        "multiple_partitioned_tables",
        [{"data": {"table1": df, "table2": different_df}}],
        partition_on="B",
    )

    sorted(dm.partitions.keys())


As noted above, when data is appended to a dataset, Kartothek guarantees it has
the proper schema and partitioning.

The order of columns provided in ``partition_on`` is important, as the partition
structure would be different if the columns are in a different order.

.. note:: Every partition must have data for every table. An empty dataframe in this
          context is also considered as data.

.. _partitioning_dask:

Force partitioning by shuffling using Dask
------------------------------------------

By default, the partitioning logic is applied per physical input partition when
writing. In particular, this means that when calling `partition_on` on a column
with total N unique values, this may create up to M x N files, where M is the
number of physical input partitions.

.. ipython:: python
    :okwarning:

    import dask.dataframe as dd
    import numpy as np
    from kartothek.io.dask.dataframe import update_dataset_from_ddf

    df = pd.DataFrame(
        {
            # Good partition column since there are only two unique values
            "A": [0, 1] * 100,
            # Too many values for partitioning but still discriminative for querying
            "B": np.repeat(range(20), 10),
            "C": "some_payload",
        }
    )

    ddf = dd.from_pandas(df, npartitions=10)

    dm = update_dataset_from_ddf(
        ddf, dataset_uuid="no_shuffle", store=store_url, partition_on="A", table="table"
    ).compute()
    sorted(dm.partitions.keys())

.. _shuffling:

Shuffling
*********

To circumvent the heavy file fragmentation, we offer a shuffle implementation
for dask dataframes which causes the fragmented files for the respective
partitioning values of A to be fused into a single file.

.. ipython:: python
    :okwarning:

    dm = update_dataset_from_ddf(
        ddf,
        dataset_uuid="with_shuffle",
        store=store_url,
        partition_on="A",
        shuffle=True,
        table="table",
    ).compute()
    sorted(dm.partitions.keys())

.. warning::

    This may require a lot of memory since we need to shuffle the data. Most of
    this increased memory usage can be compensated by using dask
    `spill-to-disk`_. If peak memory usage is an issue and needs to be
    controlled, it may be helpful to reduce the final file sizes because the
    serialization part into the Apache Parquet file format usually requires a
    bit more memory than the shuffling tasks themselves, see also
    :ref:`bucketing`.


.. _bucketing:

Bucketing
*********

If you need more control over the size of files and the distribution within the files you can also ask for explicit bucketing of values.

.. note::

    There are many reasons for wanting smaller files. One reason could be a
    reduced peak memory usage during dataset creation, another might be due to
    memory or performance requirements in later steps. If you intend to optimize
    your pipelines by reducing file sizes we also recommend to look into
    predicate pushdown, see also :ref:`efficient_querying` which might offer
    similar, synergetic effects.

Bucketing uses the values of the requested columns and assigns every unique
tuple to one of `num_buckets` files. This not only helps to control output file
sizes but also allows for very efficient querying in combination with seconday
indices, see also :ref:`efficient_querying`.

In the below example you can see the same data being used as above but this time we will bucket by column `B` which will no longer create a single file per value in `B` but rather `num_buckets` files.
When investigating the index, we can also see that a query for a given value in B will return exactly one file per partition key.

.. ipython:: python
    :okwarning:

    dm = update_dataset_from_ddf(
        ddf,
        dataset_uuid="with_bucketing",
        store=store_url,
        partition_on="A",
        shuffle=True,
        table="table",
        bucket_by="B",
        num_buckets=4,
        secondary_indices="B",
    ).compute()
    sorted(dm.partitions.keys())

    dm = dm.load_index("B", store_url)

    sorted(dm.indices["B"].eval_operator("==", 1))


.. _spill-to-disk: https://distributed.dask.org/en/latest/worker.html#memory-management
