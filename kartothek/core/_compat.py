# -*- coding: utf-8 -*-
from distutils.version import LooseVersion

import pyarrow as pa
import simplejson

ARROW_LARGER_EQ_0141 = LooseVersion(pa.__version__) >= "0.14.1"
ARROW_LARGER_EQ_0150 = LooseVersion(pa.__version__) >= "0.15.0"


def load_json(buf, **kwargs):
    """
    Compatibility function to load JSON from str/bytes/unicode.

    For Python 2.7 json.loads accepts str and unicode.
    Python 3.4 only accepts str whereas 3.5+ accept bytes and str.
    """
    if isinstance(buf, bytes):
        return simplejson.loads(buf.decode("utf-8"), **kwargs)
    else:
        return simplejson.loads(buf, **kwargs)
