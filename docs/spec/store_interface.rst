.. _store_interface:

=======================
KeyValueStore Interface
=======================

All storage interaction use ``simplekv.KeyValueStore`` as an storage layer
abstraction. This allows convenient access to many different common Key-Value
stores (ABS, S3, GCS, local filesystem, etc.) and allows an easy switch between
the storage backends to facilitate a simpler test setup.

Generally, all of our public functions accepting a ``store`` argument accept a
multitude of different input types and we generally accept all kinds of stores
inheriting from ``KeyValueStore``, assuming they implement the pickle protocol.
However, there are storages which simply cannot be distributed across processes
or network nodes sensibly. A prime Example is the ``simplekv.memory.DictStore``
which uses a simple python dictionary as a backend store. It is technically
possible to (de-)serialize the store but once it is deserialized in another
process, or another node, the store looses its meaning since the stores are
isolated per process, node, etc. Kartothek does not verify semantics of a given
store but only verifies whether or not the store implements the pickle protocol.

For all cases where the ``KeyValueStore`` does not implement the pickle
protocol, or some more complex logic is required to initialize it, kartothek
also accepts _factories_ which must be a callable returning a ``KeyValueStore``
(see also ``kartothek.core.typing.StoreFactory``).

For convenience we also offer a `storefact`_ integration and accept store urls
which proves another easy level of access and is well suited for ad-hoc
investigations.

.. _simplekv: https://simplekv.readthedocs.io/
.. _storefact: https://storefact.readthedocs.io/