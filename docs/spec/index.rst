.. _dataset_spec:

=============
Specification
=============

Rationale
=========

Storing data distributed over multiple files in an object store
(S3, ABS, GCS, etc.) allows for a fast, cost efficient and highly
scalable data infrastructure. A downside of storing data simply in
an object store is that the storages themselves give little to no
guarantees beyond the consistency of a single file. In particular,
they cannot guarantee the consistency of your *dataset*. If we demand
a consistent state of our dataset at all times we need to do track the
state of the dataset ourself. Explicit state tracking can be more than
a nuisance though, if done correctly.

Goals
=====
 * Consistent dataset state at all times
 * Dataset state is only modified by atomic commits
 * Strongly typed table schemas
 * Query planing with `O(1)` calls to the remote store
 * Inverted indices for fast query planing
 * Read access without any locking mechanism
 * Portable across frameworks and languages
 * Seemless integration to OSS community software
   (Apache Arrow, Apache Parquet, pandas, etc.)
 * Lifecycle management (garbage collection, retention, etc.)
 * No external service required for state tracking

Dataset state tracking
======================

The dataset state is tracked along with additional metadata in a single file
which allows for the implementation of atomic commits using copy-on-write
principles.
The dataset metadata is stored in either a plain JSON file or as a zstd
compressed msgpack file.

The dataset state is fully defined by:

1. List of all physical partitions
2. Schema specification for each table (`_common_metadata`)
3. Secondary (inverted) indices

Minimal Example::

    <UUID>.by-dataset-metadata.{json|msgpack.zstd}
    <UUID>/indices/<INDEX_COLUMN_NAME>/<ISOTIMESTAMP>.by-dataset-index.parquet
    <UUID>/<TABLE_NAME>/_common_metadata
    <UUID>/<TABLE_NAME>/{PARTITION_KEY=PARTITION_VALUE}/<PARTITION_UUID>.<FORMAT_SUFFIX>


Filenames
---------

Multiple datasets may reside in a single storage location; a given dataset
always resides in a single storage location. Also there will be files
in this storage location that are not part of any dataset. To identify a
dataset, the main metadata file must follow the naming specification. All
other files belonging to the dataset must be referenced by this file.

A storage location can be one of:

 * An object store bucket incl. all of its keys (S3, ABS, GCS, etc.)
 * A directory on a filesystem incl. all files in subdirectories

.. warning::

    Files cannot be shared between datasets, a dataset expects that all files
    mentioned in the metadata belong exclusively to the dataset. Thus deletion
    of all referenced files in the metadata will lead to a total deletion
    of the dataset. No other datasets will be impacted of this deletion.

General Filename rules
~~~~~~~~~~~~~~~~~~~~~~

The files (described below in detail) all consist of several forward-slash
separated components. All components must only consist of the following
characters:

 * Uppercase and lowercase English letters (a-z, A-Z)
 * Digits 0 to 9
 * Characters plus, minus and underscore (+  -  _)


Main metadata file
~~~~~~~~~~~~~~~~~~~

The filename of the main metadata file consists of the following
components that are separated by a single dot each:

 * The UUID of the dataset. This should be a string that hasn't been used yet
   for prefixing any other file in the storage location. One may use the
   Python function :func:`~kartothek.core.uuid.gen_uuid()` (UUID type 4) to generate
   such a UUID but the only requirement here is that there is no other file in
   the target location that has the same prefix.
 * The identifier ``by-dataset-metadata``. The character sequence
   ``by-dataset-metadata``. must not be used in any other file
   than main metadata files for datasets.
 * The suffix json or msgpack.zstd to describe the file format
   (msgpack.zstd denotes a zstd compressed msgpack file in a
   Kartothek dataset).

Example:
::

    0d6de3c6-b7a4-11e6-8ed1-08002753cf7b.by-dataset-metadata.json

.. note::

    All files that form the dataset must also start with the prefix "<uuid>".

Table schema
~~~~~~~~~~~~

The table schema information consists of thee components:

 * The UUID of the dataset. This should be a string that hasn't been used yet
   for prefixing any other file in the storage location. One may use the
   :func:`~kartothek.core.uuid.gen_uuid()` (UUID type 4) to generate such a UUID.
 * The table identifier
 * The string _common_metadata

Example::

    0d6de3c6-b7a4-11e6-8ed1-08002753cf7b/core/_common_metadata

The data stored in ``_common_metadata`` is supposed to be an _empty_ parquet
file fully specifying the schema of the table.
For more details, see :ref:`type_system`.


Data files of partitions
~~~~~~~~~~~~~~~~~~~~~~~~

These files must consist of the following forward-slash separated components:

 * The UUID of the dataset. This should be a string that hasn't been used yet
   for prefixing any other file in the storage location. One may use the
   :func:`~kartothek.core.uuid.gen_uuid()` (UUID type 4) to generate such a UUID.
 * The table identifier
 * (optional) partition content encoding
 * The partition identifier.
 * The suffix to describe the file format, e.g. parquet, csv, h5, etc.
   For available serialization formats, see :ref:`dataframe_serialization`

Example::

    0d6de3c6-b7a4-11e6-8ed1-08002753cf7b/core/partition_key=partition_value/part_1.parquet


.. note::

    **Partition content encoding**

    Just like Dask, Apache Spark or Apache Hive are doing, it is possible
    to encode the content of a particular column in the filename which allows
    the construction of an index based on that column. Both the column name
    and value are URL encoded and the column type is stored in the table schema
    information. The payload data file itself should not include this column
    any more but rather any reading client is supposed to type-safely
    reconstruct this column upon loading.
    For example the path
    ``0d6de3c6-b7a4-11e6-8ed1-08002753cf7b/location=123/product=3454/*.parquet``
    indicates that data with ``(location == 123 AND product == 3454)``
    is stored in this directory.

Index files
~~~~~~~~~~~

These files must consist of the following dot-separated components:

 * The UUID of the dataset. This should be a string that hasn't been used yet
   for prefixing any other file in the storage location. One may use the
   :func:`~kartothek.core.uuid.gen_uuid()` (UUID type 4) to generate such a UUID.
 * A hard coded identifier ``indices``
 * The name of the field used in the index
 * A url encoded ISO 8601 timestamp (format ``YYYY-MM-DDTHH:MM:SS.ffffff``)
 * The suffix parquet to describe the file format.

Example::

    0d6de3c6-b7a4-11e6-8ed1-08002753cf7b/indices/<FIELD_NAME>/<ISOTIMESTAMP>.by-dataset-index.parquet

Attributes
----------

This section describes the attributes that should be present in the main
metadata JSON file. For each attribute, we specify its key and the expected
type. The type is a must and conversion from e.g. ``INT`` in the case
a ``STRING`` is expected are not done. The usage of these attributes
can be seen in the example below.

 * ``dataset_metadata_version (INT) = 4``: The version of the metadata,
   needs to be increased on every specification change.
 * ``dataset_uuid (STRING)``: Unique identifier of the dataset. This needs
   to be the same as used in the filename.
 * ``metadata (MAP<STRING, STRING>)``: Arbitrary metadata that can be used
   to annotate a dataset. This may be empty or omitted.
 * ``partitions (MAP<STRING, ...>)``: Labeled set of partitions. The key is
   the partition identifier as used in the file name and in indices.
 * ``files (MAP<STRING, STRING>)``: Labeled files contained in a partition.

   * The filename must end with a known file extension, e.g. ``.parquet``.
   * All partitions shall have the same set of keys.
   * A single file must be part of exactly one dataset.

 * ``indices (MAP<STRING, STRING>)``:

   * (Secondary) indices are optional, so this mapping can be empty or
     omitted completely.
   * Indices provide support to find the matching partitions for a row
     selection. In the first iteration, an index can be used to find the set
     of matching files for a row selection with the constraint on a single
     column value (e.g. ``product_id = 12345``). For a row selection with
     multiple row constraints, one shall query all 1-column indices and use
     the intersection of the all returned partition sets.
   * The key of the map is the field on which the row selection constraint
     is defined. This field may also be a field that is not contained in the
     actual data in the case that this field would have the same value for
     all rows in a partition.
   * The value of the indices map is the name of the Parquet file storing the
     index.
   * For a storage specification of the indices, see :ref:`partition_indices`
