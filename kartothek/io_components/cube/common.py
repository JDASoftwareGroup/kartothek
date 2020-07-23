"""
Common utilities used by all IO operations.
"""
import uuid

__all__ = ("assert_stores_different", "check_blocksize", "check_store_factory")


def check_store_factory(store):
    """
    Check that given store is a factory.

    Parameters
    ----------
    store: Any
        Store passed by the user.

    Raises
    ------
    TypeError: In case the store is not a factory.
    """
    if not callable(store):
        raise TypeError(
            "store must be a factory but is {}".format(type(store).__name__)
        )


def assert_stores_different(store1, store2, prefix):
    """
    Check that given stores are different.

    This is a workaround for tha fact that simplekv stores normally do not implemenent some sane equality check.

    Parameters
    ----------
    store1: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        First store.
    store2: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Second store, will be used to write a test key to.
    prefix: str
        Prefix to be used for the temporary key used for the equality check.

    Raises
    ------
    ValueError: If stores are considered to be identical.
    """
    if callable(store1):
        store1 = store1()
    if callable(store2):
        store2 = store2()

    key = "{prefix}/.test_store_difference.{uuid}".format(
        prefix=prefix, uuid=uuid.uuid4().hex
    )
    try:
        store2.put(key, b"")
        try:
            store1.get(key)
            raise ValueError("Stores are identical but should not be.")
        except KeyError:
            pass
    finally:
        try:
            store2.delete(key)
        except KeyError:
            pass


def check_blocksize(blocksize):
    """
    Check that given blocksize is a positive integer.

    Parameters
    ----------
    blocksize: Any
        Blocksize passed by the user.

    Raises
    ------
    TypeError: In case the blocksize is not an integer.
    ValueError: In case the blocksize is < 0.
    """
    if not isinstance(blocksize, int):
        raise TypeError(
            "blocksize must be an integer but is {}".format(type(blocksize).__name__)
        )
    if blocksize <= 0:
        raise ValueError("blocksize must be > 0 but is {}".format(blocksize))
