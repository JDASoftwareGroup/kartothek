=========
Changelog
=========


Version 3.1.0 (2019-XX-XX)
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

**Breaking:**

- categorical normalization was moved from :meth:`~kartothek.core.common_metadata.make_meta` to
  :meth:`~kartothek.core.common_metadata.normalize_type`.
- :meth:`kartothek.core.common_metadata.SchemaWrapper.origin` is now a set of of strings instead of a single string


Version 3.0.0 (2019-05-02)
==========================

- Initial public release
