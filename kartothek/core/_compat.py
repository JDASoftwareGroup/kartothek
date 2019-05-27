# -*- coding: utf-8 -*-
from distutils.version import LooseVersion

import dask
import pyarrow as pa
import simplejson

ARROW_LARGER_EQ_0130 = LooseVersion(pa.__version__) >= "0.13.0"
DASK_LARGER_EQ_121 = LooseVersion(dask.__version__) >= "1.2.1"


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
