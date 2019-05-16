=========
Changelog
=========


Version 3.1.0 (2019-XX-XX)
==========================

- fix ``getargspec`` ``DeprecationWarning``
- fix ``FutureWarning`` in ``filter_array_like``
- remove ``funcsigs`` requirement
- Implement reference ``io.eager`` implementation, adding the functions:
    + ``io.eager.garbage_collect_dataset``
    + ``io.eager.index.build_dataset_indices``
    + ``io.eager.update_dataset_from_dataframes``
- fix ``_apply_partition_key_predicates`` ``FutureWarning``

**Breaking:**

- categorical normalization was moved from :meth:`~kartothek.core.common_metadata.make_meta` to
  :meth:`~kartothek.core.common_metadata.normalize_type`.


Version 3.0.0 (2019-05-02)
==========================

- Initial public release
