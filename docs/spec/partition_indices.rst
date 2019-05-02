.. _partition_indices:

=================
Partition indices
=================

TODO

A secondary index is persisted as a Parquet file with the following
(Parquet) schema:
The fieldname corresponds to the name of the column in the persisted
DataFrame.
The partition is a list of partition identifiers, as used in the keys of
the partitions map and the data filename. (Note: the partition identifier
is used instead of the data filename as a single partition can span multiple
files containing different column sets using the same row selection.)