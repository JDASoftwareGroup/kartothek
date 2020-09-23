import pickle
from functools import partial
from typing import cast

from simplekv import KeyValueStore
from storefact import get_store_from_url

from kartothek.core.naming import MAX_METADATA_VERSION, MIN_METADATA_VERSION
from kartothek.core.typing import StoreFactory, StoreInput


def _verify_metadata_version(metadata_version):
    """
    This is factored out to be an easier target for mocking
    """
    if metadata_version < MIN_METADATA_VERSION:
        raise NotImplementedError(
            "Minimal supported metadata version is 4. You requested {metadata_version} instead.".format(
                metadata_version=metadata_version
            )
        )
    elif metadata_version > MAX_METADATA_VERSION:
        raise NotImplementedError(
            "Future metadata version `{}` encountered.".format(metadata_version)
        )


def verify_metadata_version(*args, **kwargs):
    return _verify_metadata_version(*args, **kwargs)


def ensure_string_type(obj):
    """
    Parse object passed to the function to `str`.

    If the object is of type `bytes`, it is decoded, otherwise a generic string representation of the object is
    returned.

    Parameters
    ----------
    obj: Any
        object which is to be parsed to `str`

    Returns
    -------
    str_obj: String
    """
    if isinstance(obj, bytes):
        return obj.decode()
    else:
        return str(obj)


def ensure_store(store: StoreInput) -> KeyValueStore:
    """
    Ensure that the input is a valid KeyValueStore.
    """
    # This function is often used in an eager context where we may allow non-serializable stores, i.e. we do not need to check for serializability
    if isinstance(store, KeyValueStore):
        return store
    return lazy_store(store)()


def _factory_from_store(pickled_store):
    """
    Ensure that we're using internally only deserialized stores / proper
    factories to avoid funny business with unix sockets in forked subprocesses.
    """
    return pickle.loads(pickled_store)


def lazy_store(store: StoreInput) -> StoreFactory:
    """
    Create a store factory from the input. Acceptable inputs are
    * Storefact store url
    * Callable[[], KeyValueStore]
    * KeyValueStore

    If a KeyValueStore is provided, it is verified that the store is serializable
    """
    if callable(store):
        return cast(StoreFactory, store)
    elif isinstance(store, KeyValueStore):
        try:
            pickled_store = pickle.dumps(store)
        except Exception as exc:
            raise TypeError(
                """KeyValueStore not serializable.
Please consult https://kartothek.readthedocs.io/en/stable/spec/store_interface.html for more information."""
            ) from exc
        return partial(_factory_from_store, pickled_store)
    elif isinstance(store, str):
        ret_val = partial(get_store_from_url, store)
        ret_val = cast(StoreFactory, ret_val)  # type: ignore
        return ret_val
    else:
        raise TypeError(
            f"Provided incompatible store type. Got {type(store)} but expected {StoreInput}."
        )
