.. _indexing:

Indexing
========

Kartothek uses different types of `inverted file indices`_ to enable efficient partition pruning and improve query performance, see also :doc:`efficient_querying` for more hints on how to optimize performance. This section describes the different types of indices, how to create them and how to interact with them


Principle in-memory representation
----------------------------------

All currently supported kartothek index types are inverted indices and are mapping observed values of a given field to a list of partitions where they were observed.

.. ipython:: python

    index_dct = {1: ["table/12345"], 2: ["table/12345", "table/6789"]}

Where, in this example, the value ``42`` is found in exactly one partition which is labeled ``table/partitionA=42/12345``.

Users typically do not interact with indices directly since querying a dataset will automatically load and interact with the indices. For some applications it is still quite useful to interact with them directly.

All indices implement :class:`~kartothek.core.index.IndexBase` which allows the user to interact with the indices in some useful ways.

.. ipython:: python

    from kartothek.api.dataset import IndexBase

    index = IndexBase(column="FieldName", index_dct=index_dct)

    index.dtype

    index.observed_values()

    index.eval_operator(">=", 2)

    index.as_flat_series()


Partition Indices
-----------------

The first index type kartothek offers is a partition index. The partition index is created by partitioning a dataset in a hive-like partition scheme.

.. ipython:: python
    :suppress:

    import string
    import pandas as pd
    from kartothek.api.dataset import ensure_store

    store = ensure_store("hmemory://")
    from kartothek.api.dataset import store_dataframes_as_dataset

.. ipython:: python

    df = pd.DataFrame(
        {
            "PartField": ["A"] * 5 + ["B"] * 5,
            "IndexedField": list(range(5)) + list(range(3, 8)),
            "Payload": [string.ascii_letters[i] for i in range(10)],
        }
    )
    dm = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="indexing_docs",
        dfs=[df],
        partition_on=["PartField"],
        secondary_indices=["IndexedField"],
    ).load_all_indices(store)

    sorted(store.keys())

    part_index = dm.indices["PartField"]
    part_index

This kind of index is also called a `primary index`. This implies the property that a given file is guaranteed to only contain **one** unique value of the given field. This can also be observed when investigating the flat structure of the index.

.. ipython:: python

    part_index.as_flat_series()

This property makes this kind of index very powerful if used correctly since it prunes the partitions exactly to the user query and enables exact removal of data when mutating datasets (see :doc:`../guide/mutating_datasets`).

For data with high cardinality this kind of index is not well suited since it would result in a highly fragmented dataset with too many, too small files.


Secondary indices
-----------------

Secondary indices are the most powerful type of indices which allow us to reference files without having to encode any kind of values in the keys. They can be created by supplying the `secondary_indices` keyword argument as shown above. The user interaction works similarly to the


Persistence
~~~~~~~~~~~

A secondary index is persisted as a Parquet file with the following
(Parquet) schema:
The field name corresponds to the name of the column in the persisted
DataFrame.
The partition is a list of partition identifiers, as used in the keys of
the partitions map and the data filename. (Note: the partition identifier
is used instead of the data filename as a single partition can span multiple
files containing different column sets using the same row selection.)


Typing
------

Every index has a well defined arrow data type which is usually inferred automatically and ensured to be consistent with the overall dataset schema.

.. ipython:: python

    part_index.dtype


Supported data types for indices include

* ``bool``
* ``(u)int{8,16,32,64}``
* ``float{32,64}``
* ``str``
* ``bytes``
* ``pd.Timestamp`` (with and without timezones)
* ``datetime.date``

.. important::

    Nullable fields are not properly supported and depending on the used API, the behaviour may be slightly different.

    In particular, the plain dataset API will usually drop nan/nulls silently while the Cube API will raise an exception.


See also
--------
* :doc:`efficient_querying` for some general hints for faster quering
* :doc:`storage_layout`
* :doc:`../guide/partitioning` for some guidance on how to partition a dataset
* :doc:`../guide/dask_indexing`



.. _inverted file indices: https://en.wikipedia.org/wiki/Inverted_index
