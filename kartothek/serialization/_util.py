# -*- coding: utf-8 -*-
import six


def _check_contains_null(val):
    if isinstance(val, six.binary_type):
        for byte in val:
            if isinstance(byte, six.binary_type):
                compare_to = chr(0)
            else:
                compare_to = 0
            if byte == compare_to:
                return True
    return False


def ensure_unicode_string_type(obj):
    """
    ensures obj is a of native string type:
    python 2: ``unicode``, python 3 ``str`` (aka ``six.text_type``)
    """
    if isinstance(obj, six.binary_type):
        return obj.decode("utf8")
    else:
        return six.text_type(obj)
