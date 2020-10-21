.. _efficient_querying:

==================
Efficient Querying
==================

In general, Kartothek should know as early as possible which data to exclude, so
Kartothek doesn't have to read and transfer it. If you're going to query a
dataset a lot, you should consider storing it in an optimized layout. See
:ref:`writing`.


.. _querying_process:

Background
----------

To understand how to query efficiently, you should understand Kartothek's
querying process:

When querying data, Kartothek tries to apply the given predicates as early as
possible, so it has to read and parse as few parquet files as possible:

- During partition pruning, Kartothek looks if there exists an index for the
  filtered column. This may either be if we partitioned the column (primary
  index) or if we generated a secondary index for it. Kartothek then combines
  the results for multiple columns.
- In the resulting set of Parquet files, Kartothek looks at the Parquet
  metadata, which states the range of values for each column in this file. If
  there are multiple row groups, each one specifies the range of values inside
  it. Kartothek checks if the queried data is in the range of a row group and
  therefore if this row group is loaded. This process is called Predicate
  Pushdown.

Kartothek then fetches these row groups and filters the resulting Pandas
dataframes for the conditions that could not be (completely) resolved by
then.

You can find more information on Parquet in the `Parquet Documentation
<https://parquet.apache.org/documentation/latest/>`_.


Reading
-------

Following the dataset's :ref:`storage_layout` and querying process you should
include the following in your query:

* restrictions on indexed columns (columns on which the dataset is partitioned
  or secondary indices)
* specific selection of columns

Through restrictions on partitioned columns and indexed columns, Kartothek can
significantly reduce the number of parquet files to look at. The selection of
columns allows it to transfer only a subset of the data.

.. warning::
   When using the `in` operation to filter your data, avoid passing a long list
   of values (i.e. larger than 100 elements in the list), as this might slow
   down the read performance.

Example
-------

In this example, our data has three integer columns ``A``, ``B`` and ``C``. The
dataset is partitioned by ``A`` and a secondary index for ``B`` was generated.

.. code-block:: python

    df = kartothek.io.eager.read_table(
        store=some_store,
        predicates=[[("A", ">=", 5), ("A", "<", 10), ("B", "==", 100), ("C", "<", 0)]],
    )

- Kartothek looks into the index for ``B`` to find the relevant Parquet files.
- It filters the Parquet files for the ones, that lie in the partitions ``A=5``
  to ``A=9``.
- The Parquet files are filtered by looking at their footer. Kartothek can see
  there if the range of values for column ``C`` match the searched values.
- The relevant files are loaded and filtered with Pandas for ``C < 0``.

Details on the parallel querying process can be found in :ref:`parallel_dask`.


.. _writing:

Writing
-------

When writing datasets, you should think about the following points to optimize
for better querying speed. When in doubt, test your chosen parameters and
refine them based on your results.

* Generate partitions and indices appropriate for your use case.
* Look for the right sizing of the parquet files.
* Check your row group sizing. They allow for partial loading of a Parquet file.
* Sort column contents to produce compact and disjoint value ranges between row
  groups. This allows for better selections of row groups. 
* Avoid string columns. Try to use a more specific datatype. See
  :doc:`type_system` for more information on possible column types.

See :func:`kartothek.io.dask.dataframe.update_dataset_from_ddf` for more
information.