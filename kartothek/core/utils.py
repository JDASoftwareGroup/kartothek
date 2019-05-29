from kartothek.core.naming import MAX_METADATA_VERSION, MIN_METADATA_VERSION


def _check_callable(store_factory, obj_type="store"):
    if not callable(store_factory):
        raise TypeError("{} must be a factory function".format(obj_type))


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
