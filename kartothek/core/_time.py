"""
Functions to get the current time.

kartothek modules that need to get the current time should do so only
via this module. This allows test to monkeypatch the methods
in this module to fake certain fixed times.
"""

import datetime


def datetime_now():
    """
    Get the current time as datimetime object

    Same as datetime.datetime.now
    """
    return datetime.datetime.now()


def datetime_utcnow():
    """
    Get the current time as datimetime object

    Same as datetime.datetime.utcnow
    """
    return datetime.datetime.utcnow()
