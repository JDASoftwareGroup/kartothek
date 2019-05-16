=========
Changelog
=========


Version 3.1.0 (2019-XX-XX)
==========================

- Implement reference ``io.eager`` implementation, adding the functions:
    + ``io.eager.garbage_collect_dataset``
    + ``io.eager.index.build_dataset_indices``
    + ``io.eager.update_dataset_from_dataframes``


Version 3.0.1 (2019-XX-XX)
==========================

- fix ``getargspec`` ``DeprecationWarning``
- fix ``FutureWarning`` in ``filter_array_like``
- remove ``funcsigs`` requirement
- fix ``_apply_partition_key_predicates`` ``FutureWarning``


Version 3.0.0 (2019-05-02)
==========================

- Initial public release
