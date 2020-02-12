=========
Changelog
=========

Version 3.7.0 (2020-02-12)
==========================

Improvements
^^^^^^^^^^^^

* Support for pyarrow 0.16.0
* Decrease scheduling overhead for dask based pipelines
* Performance improvements for categorical data when using pyarrow>=0.15.0
* Dask is now able to calculate better size estimates for the following classes:
    * :class:`~kartothek.core.dataset.DatasetMetadata`
    * :class:`~kartothek.core.factory.DatasetFactory`
    * :class:`~kartothek.io_components.metapartition.MetaPartition`
    * :class:`~kartothek.core.index.ExplicitSecondaryIndex`
    * :class:`~kartothek.core.index.PartitionIndex`
    * :class:`~kartothek.core.partition.Partition`
    * :class:`~kartothek.core.common_metadata.SchemaWrapper`


Version 3.6.2 (2019-12-17)
==========================

Improvements
^^^^^^^^^^^^

* Add more explicit typing to :mod:`kartothek.io.eager`.

Bug fixes
^^^^^^^^^
* Fix an issue where :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf` would create a column named "_KTK_HASH_BUCKET" in the dataset


Version 3.6.1 (2019-12-11)
==========================

Bug fixes
^^^^^^^^^
* Fix a regression introduced in 3.5.0 where predicates which allow multiple
  values for a field would generate duplicates

Version 3.6.0 (2019-12-03)
==========================

New functionality
^^^^^^^^^^^^^^^^^
- The partition on shuffle algorithm in :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf` now supports
  producing deterministic buckets based on hashed input data.

Bug fixes
^^^^^^^^^
- Fix addition of bogus index columns to Parquet files when using `sort_partitions_by`.
- Fix bug where ``partition_on`` in write path drops empty DataFrames and can lead to datasets without tables.


Version 3.5.1 (2019-10-25)
==========================
- Fix potential ``pyarrow.lib.ArrowNotImplementedError`` when trying to store or pickle empty
  :class:`~kartothek.core.index.ExplicitSecondaryIndex` objects
- Fix pickling of :class:`~kartothek.core.index.ExplicitSecondaryIndex` unloaded in
  :func:`~kartothek.io_components.read.dispatch_metapartitions_from_factory`


Version 3.5.0 (2019-10-21)
==========================

New functionality
^^^^^^^^^^^^^^^^^
- Add support for pyarrow 0.15.0
- Additional functions in `kartothek.serialization` module for dealing with predicates
  * :func:`~kartothek.serialization.check_predicates`
  * :func:`~kartothek.serialization.filter_predicates_by_column`
  * :func:`~kartothek.serialization.columns_in_predicates`
- Added available types for type annotation when dealing with predicates
  * `~kartothek.serialization.PredicatesType`
  * `~kartothek.serialization.ConjunctionType`
  * `~kartothek.serialization.LiteralType`
- Make ``kartothek.io.*read_table*`` methods use default table name if unspecified
- ``MetaPartition.parse_input_to_metapartition`` accepts dicts and list of tuples equivalents as ``obj`` input
- Added `secondary_indices` as a default argument to the `write` pipelines

Bug fixes
^^^^^^^^^
- Input to ``normalize_args`` is properly normalized to ``list``
- ``MetaPartition.load_dataframes`` now raises if table in ``columns`` argument doesn't exist
- require ``urlquote>=1.1.0`` (where ``urlquote.quoting`` was introduced)
- Improve performance for some cases where predicates are used with the `in` operator.
- Correctly preserve :class:`~kartothek.core.index.ExplicitSecondaryIndex` dtype when index is empty
- Fixed DeprecationWarning in pandas ``CategoricalDtype``
- Fixed broken docstring for `store_dataframes_as_dataset`
- Internal operations no longer perform schema validations. This will improve
  performance for batched partition operations (e.g. `partition_on`) but will
  defer the validation in case of inconsistencies to the final commit. Exception
  messages will be less verbose in these cases as before.
- Fix an issue where an empty dataframe of a partition in a multi-table dataset
  would raise a schema validation exception
- Fix an issue where the `dispatch_by` keyword would disable partition pruning
- Creating dataset with non existing columns as explicit index to raise a ValueError

Breaking changes
^^^^^^^^^^^^^^^^
- Remove support for pyarrow < 0.13.0
- Move the docs module from `io_components` to `core`


Version 3.4.0 (2019-09-17)
==========================
- Add support for pyarrow 0.14.1
- Use urlquote for faster quoting/unquoting


Version 3.3.0 (2019-08-15)
==========================
- Fix rejection of bool predicates in :func:`~kartothek.serialization.filter_array_like` when bool columns contains
  ``None``
- Streamline behavior of `store_dataset_from_ddf` when passing empty ddf.
- Fix an issue where a segmentation fault may be raised when comparing MetaPartition instances
- Expose a ``date_as_object`` flag in ``kartothek.core.index.as_flat_series``


Version 3.2.0 (2019-07-25)
==========================
- Fix gh:66 where predicate pushdown may evalute false results if evaluated
  using improper types. The behavior now is to raise in these situations.
- Predicate pushdown and :func:`~kartothek.serialization.filter_array_like` will now properly handle pandas Categoricals.
- Make the size of partitions controllable by introducing the `partition_size` parameter in `kartothek.io.dask.bag.read_dataset_as_metapartitions_bag` and `karothek.io.dask.bag.read_dataset_as_dataframes_bag`
- Add :meth:`~karothek.io.dask.bag.read_dataset_as_dataframes_bag`
- Add :meth:`~karothek.io.dask.bag.read_dataset_as_metapartitions_bag`


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
