# -*- coding: utf-8 -*-


def _check_contains_null(val):
    if isinstance(val, bytes):
        for byte in val:
            if isinstance(byte, bytes):
                compare_to = chr(0)
            else:
                compare_to = 0
            if byte == compare_to:
                return True
    return False


def ensure_unicode_string_type(obj):
    """
    ensures obj is a of native string type:
    """
    if isinstance(obj, bytes):
        return obj.decode("utf8")
    else:
        return str(obj)
