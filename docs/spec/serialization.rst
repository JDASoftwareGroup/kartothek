.. _dataframe_serialization:

=======================
DataFrame Serialization
=======================

Serialise Pandas DataFrames to/from bytes

Serialisation to bytes
----------------------

For the serialsation, we need to pick a format serialiser, you either use
:func:`~kartothek.serialization.default_serializer` or explicitly select a serialiser,
e.g. :class:`~kartothek.serialization.ParquetSerializer`.

.. code:: python

    from kartothek.serialization import ParquetSerializer

    serialiser = ParquetSerializer()
    df = ...
    serialiser.store(store, "storage_key", df)


Deserialisation
---------------

For deserialisation, you don't have to instantiate any serialiser as the correct
one is determined from the filename.

.. code:: python

    from kartothek.serialization import DataFrameSerializer

    df = DataFrameSerializer.restore_dataframe(store, "file.parquet")
    # You can also supply a filter on the loaded DataFrame, e.g.
    df = DataFrameSerializer.restore_dataframe(store, "file.parquet", "c_id > 42000")
    # Currently these filter queries are passed to pandas.DataFrame.query but in
    # future they could be further passed on to the file format depening on if
    # the format supports predicate pushdown (currently this is only Parquet)
