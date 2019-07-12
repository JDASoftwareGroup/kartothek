=========
Changelog
=========


Version 3.1.1 (2019-07-12)
==========================

- make :meth:`~karothek.io.dask.bag.build_dataset_indices__bag` more efficient
- make :meth:`~kartothek.io.eager.build_dataset_indices` more efficient
- fix pseudo-private :meth:`~kartothek.io_components.read.dispatch_metapartitions` handling of
  ``concat_partitions_on_primary_index``
- fix internal errors if querying (e.g. via :meth:`~kartothek.io.eager.read_dataset_as_dataframes`) with
  ``datetime.date`` predicates that use the dataset index; this affects all code paths using
  :meth:`~kartothek.io_components.metapartition.MetaPartition.load_dataframes`


Version 3.1.0 (2019-07-10)
==========================

- fix ``getargspec`` ``DeprecationWarning``
- fix ``FutureWarning`` in ``filter_array_like``
- remove ``funcsigs`` requirement
- Implement reference ``io.eager`` implementation, adding the functions:

    - :meth:`~kartothek.io.eager.garbage_collect_dataset`
    - :meth:`~kartothek.io.eager.build_dataset_indices`
    - :meth:`~kartothek.io.eager.update_dataset_from_dataframes`

- fix ``_apply_partition_key_predicates`` ``FutureWarning``
- serialize :class:`~kartothek.core.index.ExplicitSecondaryIndex` to parquet
- improve messages for schema violation errors
- Ensure binary column names are read as type ``str``:

    - Ensure dataframe columns are of type ``str`` in :func:`~kartothek.core.common_metadata.empty_dataframe_from_schema`
    - Testing: create :func:`~kartothek.io.testing.read.test_binary_column_metadata` which checks column names stored as
      ``bytes`` objects are read as type ``str``

- fix issue where it was possible to add an index to an existing dataset by using update functions and partition indices
  (https://github.com/JDASoftwareGroup/kartothek/issues/16).

- fix issue where unreferenced files were not being removed when deleting an entire dataset

- support nested :class:`~kartothek.io_components.metapartition.MetaPartition`
  in :meth:`~kartothek.io_components.metapartition.MetaPartition.add_metapartition`.
  This fixes issue https://github.com/JDASoftwareGroup/kartothek/issues/40 .

- Add :meth:`~karothek.io.dask.bag.build_dataset_indices__bag`

- Return :class:`~dask.bag.Item` object from :meth:`~kartothek.io.dask.bag.store_bag_as_dataset` to avoid misoptimization

**Breaking:**

- categorical normalization was moved from :meth:`~kartothek.core.common_metadata.make_meta` to
  :meth:`~kartothek.core.common_metadata.normalize_type`.
- :meth:`kartothek.core.common_metadata.SchemaWrapper.origin` is now a set of of strings instead of a single string
- ``Partition.from_v2_dict`` was removed, use :meth:`kartothek.core.partition.Partition.from_dict` instead


Version 3.0.0 (2019-05-02)
==========================

- Initial public release
