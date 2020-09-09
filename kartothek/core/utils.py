from functools import partial
from typing import cast

from simplekv import KeyValueStore
from storefact import get_store_from_url

from kartothek.core.naming import MAX_METADATA_VERSION, MIN_METADATA_VERSION
from kartothek.core.typing import STORE_FACTORY_TYPE, STORE_TYPE


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


def ensure_store(store: STORE_TYPE) -> KeyValueStore:
    return lazy_store(store)()


def lazy_store(store: STORE_TYPE) -> STORE_FACTORY_TYPE:
    if callable(store):
        return cast(STORE_FACTORY_TYPE, store)
    elif isinstance(store, KeyValueStore):
        return lambda: store
    elif isinstance(store, str):
        ret_val = partial(get_store_from_url, store)
        ret_val = cast(STORE_FACTORY_TYPE, ret_val)  # type: ignore
        return ret_val
    else:
        raise TypeError(
            f"Provided incompatible store type. Got {type(store)} but expected {STORE_TYPE}."
        )
