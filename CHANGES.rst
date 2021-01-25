=========
Changelog
=========


Version 3.18.0 (2021-01-25)
===========================

* Add ``cube.suppress_index_on`` to switch off the default index creation for dimension columns
* Fixed the import issue of zstd module for `kartothek.core _zmsgpack`.
* Fix a bug in :func:`~kartothek.io_components.read.dispatch_metapartitions_from_factory` where
  `dispatch_by=[]` would be treated like `dispatch_by=None`, not merging all dataset partitions into
  a single partitions.
  
Version 3.17.3 (2020-12-04)
===========================

* Allow ``pyarrow==2`` as a dependency.

Version 3.17.2 (2020-12-01)
===========================

* #378 Improve logging information for potential buffer serialization errors


Version 3.17.1 (2020-11-24)
===========================

Bugfixes
^^^^^^^^

* Fix GitHub #375 by loosening checks of the supplied store argument


Version 3.17.0 (2020-11-23)
===========================

Improvements
^^^^^^^^^^^^
* Improve performance for "in" predicate literals using long object lists as values
* :func:`~kartothek.io.eager.commit_dataset` now allows to modify the user
  metadata without adding new data.

Bugfixes
^^^^^^^^
* Fix an issue where :func:`~kartothek.io.dask.dataframe.collect_dataset_metadata` would return
  improper rowgroup statistics
* Fix an issue where :func:`~kartothek.io.dask.dataframe.collect_dataset_metadata` would execute
  ``get_parquet_metadata`` at graph construction time
* Fix a bug in :func:`kartothek.io.eager_cube.remove_partitions` where all partitions were removed
  instead of non at all.
* Fix a bug in :meth:`~kartothek.core.dataset.DatasetMetadataBase.get_indices_as_dataframe` which would
  raise an ``IndexError`` if indices were empty or had not been loaded

Version 3.16.0 (2020-09-29)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Allow filtering of nans using "==", "!=" and "in" operators

Bugfixes
^^^^^^^^
* Fix a regression which would not allow the usage of non serializable stores even when using factories


Version 3.15.1 (2020-09-28)
===========================
* Fix a packaging issue where `typing_extensions` was not properly specified as
  a requirement for python versions below 3.8

Version 3.15.0 (2020-09-28)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Add :func:`~kartothek.io.dask.dataframe.store_dataset_from_ddf` to offer write
  support of a dask dataframe without update support. This forbids or explicitly
  allows overwrites and does not update existing datasets.
* The ``sort_partitions_by`` feature now supports multiple columns. While this
  has only marginal effect for predicate pushdown, it may be used to improve the
  parquet compression.
* ``build_cube_from_dataframe`` now supports the ``shuffle`` methods offered by
  :func:`~kartothek.io.dask.dataframe.store_dataset_from_ddf` and
  :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf` but writes the
  output in the cube format

Improvements
^^^^^^^^^^^^
* Reduce memory consumption during index write.
* Allow `simplekv` stores and `storefact` URLs to be passed explicitly as input for the `store` arguments

Version 3.14.0 (2020-08-27)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Add ``hash_dataset`` functionality

Improvements
^^^^^^^^^^^^

* Expand ``pandas`` version pin to include 1.1.X
* Expand ``pyarrow`` version pin to include 1.x
* Large addition to documentation for multi dataset handling (Kartothek Cubes)

Version 3.13.1 (2020-08-04)
===========================

* Fix evaluation of "OR"-connected predicates (#295)

Version 3.13.0 (2020-07-30)
===========================

Improvements
^^^^^^^^^^^^

* Update timestamp related code into Ktk Discover Cube functionality.
* Support backward compatibility to old cubes and fix for cli entry point.

Version 3.12.0 (2020-07-23)
===========================

New functionality
^^^^^^^^^^^^^^^^^

* Introduction of ``cube`` Functionality which is made with multiple Kartothek datasets.
* Basic Features - Extend, Query, Remove(Partitions),
  Delete (can delete entire datasets/cube), API, CLI, Core and IO features.
* Advanced Features - Multi-Dataset with Single Table, Explicit physical Partitions, Seed based join system.


Version 3.11.0 (2020-07-15)
===========================

New functionality
^^^^^^^^^^^^^^^^^

* Add :meth:`~kartothek.io_components.metapartition.MetaPartition.get_parquet_metadata` and :func:`~kartothek.io.dask.dataframe.collect_dataset_metadata`, enabling users to collect information about the Parquet metadata of a dataset (#306)

Bug fixes
^^^^^^^^^

* Performance of dataset update with ``delete_scope`` significantly improved for datasets with many partitions (#308)


Version 3.10.0 (2020-07-02)
===========================

Improvements
^^^^^^^^^^^^
* Dispatch performance improved for large datasets including metadata
* Introduction of ``dispatch_metadata`` kwarg to metapartitions read pipelines
  to allow for transition for future breaking release.

Bug fixes
^^^^^^^^^

* Ensure that the empty (sentinel) DataFrame used in :func:`~kartothek.io.eager.read_table`
  also has the correct behaviour when using the ``categoricals`` argument.


Breaking changes in ``io_components.read``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``dispatch_metapartitions`` and ``dispatch_metapartitions_from_factory``
  will no longer attach index and metadata information to the created MP
  instances, unless explicitly requested.


Version 3.9.0 (2020-06-03)
==========================

Improvements
^^^^^^^^^^^^
* Arrow 0.17.X support
* Significant performance improvements for shuffle operations in
  :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf`
  for large dask.DataFrames with many payload columns by using in-memory
  compression during the shuffle operation.
* Allow calling :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf`
  without `partition_on` when `shuffle=True`.
* :func:`~kartothek.io.dask.dataframe.read_dataset_as_ddf` supports kwarg ``dispatch_by``
  to control the internal partitioning structure when creating a dataframe.
* :func:`~kartothek.io.dask.dataframe.read_dataset_as_ddf` and :func:`~kartothek.io.dask.dataframe.update_dataset_from_ddf`
  now allow the keyword ``table`` to be optional, using the default SINGLE_TABLE identifier.
  (recommended since the multi table dataset support is in sunset).


Version 3.8.2 (2020-04-09)
==========================

Improvements
^^^^^^^^^^^^

* Read performance improved for, especially for partitioned datasets and queries with empty payload columns.

Bug fixes
^^^^^^^^^
* GH262: Raise an exception when trying to partition on a column with null values to prevent silent data loss
* Fix multiple index creation issues (cutting data, crashing) for ``uint`` data
* Fix index update issues for some types resulting in ``TypeError: Trying to update an index with different types...``
  messages.
* Fix issues where index creation with empty partitions can lead to ``ValueError: Trying to create non-typesafe index``


Version 3.8.1 (2020-03-20)
==========================

Improvements
^^^^^^^^^^^^

* Only fix column odering when restoring ``DataFrame`` if the ordering is incorrect.

Bug fixes
^^^^^^^^^
* GH248 Fix an issue causing a ValueError to be raised when using `dask_index_on` on non-integer columns
* GH255 Fix an issue causing the python interpreter to shut down when reading an
  empty file (see also https://issues.apache.org/jira/browse/ARROW-8142)

Version 3.8.0 (2020-03-12)
==========================

Improvements
^^^^^^^^^^^^

* Add keyword argument `dask_index_on` which reconstructs a dask index from an kartothek index when loading the dataset
* Add method :func:`~kartothek.core.index.IndexBase.observed_values` which returns an array of all observed values of the index column
* Updated and improved documentation w.r.t. guides and API documentation

Bug fixes
^^^^^^^^^
* GH227 Fix a Type error when loading categorical data in dask without
  specifying it explicitly
* No longer trigger the SettingWithCopyWarning when using bucketing
* GH228 Fix an issue where empty header creation from a pyarrow schema would not
  normalize the schema which causes schema violations during update.
* Fix an issue where :func:`~kartothek.io.eager.create_empty_dataset_header`
  would not accept a store factory.


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
