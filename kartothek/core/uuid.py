"""
UUID generation mechanism used in Kartothek.

Using these routines ensures compatibility w/ Kartothek as well as the application of best practices.
"""


import uuid


def gen_uuid():
    """
    Generate new UUID.

    Returns
    -------
    uuid: str
        UUID
    """
    return _uuid_hook_str()


def _uuid_hook_object():
    """
    Internal UUID function that could easily be overwritten for tests.

    Returns
    -------
    uuid: uuid.UUID
        UUID
    """
    return uuid.uuid4()


def _uuid_hook_str():
    """
    Internal UUID function that could easily be overwritten for tests.

    Returns
    -------
    uuid: str
        UUID
    """
    return _uuid_hook_object().hex


def gen_uuid_object():
    """
    Generate new UUID.

    Returns
    -------
    uuid: uuid.UUID
        UUID
    """
    return _uuid_hook_object()
