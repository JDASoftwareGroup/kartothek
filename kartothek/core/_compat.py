# -*- coding: utf-8 -*-


import simplejson
import six


def load_json(buf, **kwargs):
    """
    Compability function to load JSON from str/bytes/unicode.

    For Python 2.7 json.loads accepts str and unicode.
    Python 3.4 only accepts str whereas 3.5+ accept bytes and str.
    """
    if isinstance(buf, six.binary_type):
        return simplejson.loads(buf.decode("utf-8"), **kwargs)
    else:
        return simplejson.loads(buf, **kwargs)
