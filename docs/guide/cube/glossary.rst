Glossary
--------

.. glossary::

    Build
        Process of creating a new cube.

    Cell
        A unique combination of :term:`Dimension` values. Will result in a single row in input and output DataFrames.

    Cube
        A combination of multiple datasets that model an `Data Cubes`_-like construct. The core data structure of kartothek cube.

    Dataset ID
        The ID of a dataset that belongs to the cube w/o any :term:`Uuid Prefix`.

    Dimension
        Part of the address for a certain cube :term:`Cell`. Usually refered as :term:`Dimension Column`. Different
        dimension should describe orthogonal attributes.

    Dimension Column
        DataFrame column that contains values for a certain :term:`Dimension`.

    Dimension Columns
        Ordered list of all :term:`Dimension Column` for a :term:`Cube`.

    Extend
        Process of adding new datasets to an existing cube.

    Index Column
        Column for which additional index structures are build.

    Kartothek Dataset UUID
        Name that makes a dataset unique in a store, includes :term:`Uuid Prefix` and :term:`Dataset ID` as
        ``<UUID Prefix>++<Dataset ID>``.

    Logical Partition
        Partition that was created by ``partition_by`` arguments to the :term:`Query`.

    Physical Partition
        A single chunk of data that is stored to the blob store. May contain multiple `Parquet`_ files.

    Partition Column
        DataFrame column that contains one part that makes a :term:`Physical Partition`.

    Partition Columns
        Ordered list of all :term:`Partition Column` for a :term:`Cube`.

    Projection
        Process of dimension reduction of a cube (like a 3D object projects a shadow on the wall). Only works if the
        involved payload only exists in the subdimensional space since no automatic aggregation is supported.

    Seed
        Dataset that provides the groundtruth about which :term:`Cell` are in a :term:`Cube`.

    Store Factory
        A callable that does not take any arguments and creates a new `simplekv`_ store when being called. Its type is
        ``Callable[[], simplekv.KeyValueStore]``.

    Query
        A request for data from the cube, including things like "payload columns", "conditions", and more.

    Query Execution
        Process of reading out data from a :term:`Cube`, aka the execution of a :term:`Query`.

    Query Intention
        The actual intention of a :term:`Query`, e.g.:

        - if the user queries "all columns", the intention includes the concrete set of columns
        - if the user does not specify the dimension columns, it should use the cube dimension column (aka "no
          :term:`Projection`")

    Uuid Prefix
        Common prefix for all datasets that belong to a :term:`Cube`.

.. _Data Cubes: https://en.wikipedia.org/wiki/Data_cube
.. _Parquet: https://parquet.apache.org/
.. _simplekv: https://simplekv.readthedocs.io/