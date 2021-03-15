.. _dataframe_serialization:

=======================
DataFrame Serialization
=======================

Serialise Pandas DataFrames to/from bytes

Serialisation to bytes
----------------------

For the serialsation, we need to pick a format serializer, you either use
:func:`~kartothek.serialization.default_serializer` or explicitly select a serializer,
e.g. :class:`~kartothek.serialization.ParquetSerializer`.

.. code:: python

    from kartothek.api.serialization import ParquetSerializer

    serializer = ParquetSerializer()
    df = ...
    serializer.store(store, "storage_key", df)


Deserialisation
---------------

For deserialisation, you don't have to instantiate any serializer as the correct
one is determined from the filename.

.. code:: python

    from kartothek.api.serialization import DataFrameSerializer

    df = DataFrameSerializer.restore_dataframe(store, "file.parquet")


Supported data types
--------------------

Kartothek generally does not impose any restrictions on the data types to be used as long as they are compatible and in alignment with the :doc:`pyarrow pandas integration<python/pandas>`.

For a detailed explanation about how types are handled, please consult :doc:`type_system`.

.. _predicate_pushdown:

Filtering / Predicate pushdown
------------------------------

You can provide a filter expression in a `DNF`_ in a format of a nested list where every inner list is interpreted as a logical `conjunction` (``AND``) whereas the entire expression is interpreted as one `disjunction` (``OR``)

.. code:: python

    prediactes = [
        [("ColumnA", "==", 5),],
        [("ColumnA", ">", 5), ("ColumnB", "<=", datetime.date(2021, 1, 1)),],
    ]

The above list of predicates can be interpreted as the following whereclause::

    ColumnA = 5 OR (ColumnA > 5 AND ColumnB < '2021-01-01')


The predicate expression can be provided to the `predicates` keyword argument of the serializer and/or full dataset read interfaces.


.. note::

    All kartothek reading pipelines are exposing this `predicates` argument as well where it is not only used for predicate pushdown but also for partition pruning. See :doc:`efficient_querying` for details.

    The kartothek cube interface, see :doc:`../guide/cube/kartothek_cubes`, exposes a similar mechanism via the :class:`~kartothek.core.cube.conditions.Condition` and :class:`~kartothek.core.cube.conditions.Conjunction` classes.


Literals, operators and typing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The literals used for building the predicates are tuples with three elements.

.. code::

    (<FieldName: str>, <Operator: str>, <Value: Any>)

* ``FieldName`` is a str identifying the column this literal describes.
* ``Operator`` is a string for the logical operation applied to the field. Available operators are ``==``, ``!=``, ``<=``, ``>=``, ``<``, ``>``, ``in``
* ``Value`` is the actual value for the query. The type of this value is always required to be identical to the fields data type. We apply the same type normalization for the predicates as described in :doc:`type_system`.


Filtering for missing values / nulls is supported with operators `==`, `!=` and `in` and values `np.nan` and `None` for float and string columns respectively.


See also
--------
* :class:`~kartothek.serialization.DataFrameSerializer`
* :class:`~kartothek.serialization.ParquetSerializer`
* :doc:`efficient_querying`
* :doc:`../guide/cube/query_system`
* :doc:`type_system`


.. _DNF: https://en.wikipedia.org/wiki/Disjunctive_normal_form
