from kartothek.core.naming import MAX_METADATA_VERSION, MIN_METADATA_VERSION


def _check_callable(store_factory, obj_type="store"):
    if not callable(store_factory):
        raise TypeError("{} must be a factory function".format(obj_type))


def verify_metadata_version(metadata_version):
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
