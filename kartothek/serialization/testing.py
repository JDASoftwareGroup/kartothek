# -*- coding: utf-8 -*-

from datetime import date, datetime

import numpy as np
import pandas as pd


def _to_binary(x):
    return str(x).encode("utf-8")


BINARY_COLUMNS = [
    _to_binary(chr(1)),
    b"1",
    b"2",
    b"3",
    "4".encode("utf-16"),
    "4".encode("utf-32"),
    # this is a type1 UUID
    b"\x8f\xb6\xe5@\x90\xdc\x11\xe8\xa0\xae\x02B\xac\x12\x01\x06",
    "🙈".encode("utf-8"),
    _to_binary(chr(128)),
]


def get_dataframe_not_nested(n):
    if n > len(BINARY_COLUMNS):
        n_gen = n - len(BINARY_COLUMNS)
        binaries = BINARY_COLUMNS + [
            _to_binary(x)
            for x in range(len(BINARY_COLUMNS), n_gen + len(BINARY_COLUMNS))
        ]
    else:
        binaries = BINARY_COLUMNS[:n]
    return pd.DataFrame(
        {
            "bool": pd.Series(
                [1] * int(np.floor(n / 2)) + [0] * int(np.ceil(n / 2)), dtype=np.bool
            ),
            "int8": pd.Series(range(n), dtype=np.int8),
            "int16": pd.Series(range(n), dtype=np.int16),
            "int32": pd.Series(range(n), dtype=np.int32),
            "int64": pd.Series(range(n), dtype=np.int64),
            "uint8": pd.Series(range(n), dtype=np.uint8),
            "uint16": pd.Series(range(n), dtype=np.uint16),
            "uint32": pd.Series(range(n), dtype=np.uint32),
            "uint64": pd.Series(range(n), dtype=np.uint64),
            "float32": pd.Series([float(x) for x in range(n)], dtype=np.float32),
            "float64": pd.Series([float(x) for x in range(n)], dtype=np.float64),
            "date": pd.Series(
                [date(2018, 1, x % 31 + 1) for x in range(1, n + 1)], dtype=object
            ),
            "datetime64": pd.Series(
                [datetime(2018, 1, x % 31 + 1) for x in range(1, n + 1)],
                dtype="datetime64[ns]",
            ),
            "unicode": pd.Series([str(x) for x in range(n)], dtype=np.unicode),
            "null": pd.Series([None] * n, dtype=object),
            "bytes": pd.Series(binaries, dtype=np.object),
        }
    ).sort_index(axis=1)
