#!/usr/bin/env python
import os

import pyarrow as pa
from storefact import get_store_from_url

from kartothek.core.testing import get_dataframe_alltypes
from kartothek.serialization import ParquetSerializer

if __name__ == "__main__":
    ser = ParquetSerializer()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    store = get_store_from_url(f"hfs://{dir_path}")

    df = get_dataframe_alltypes()
    df["byte"] = b"\x82\xd6\xc1\x06Z\x08\x11\xe9\x85eJ\x00\x07\xf8\n\x10"
    ref_file = f"{pa.__version__}"
    ser.store(store, ref_file, df)
