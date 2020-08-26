import logging
from functools import partial
from typing import List, Union

import dask.dataframe as dd
import pandas as pd

_logger = logging.getLogger()
_PAYLOAD_COL = "__ktk_shuffle_payload"


def pack_payload_pandas(partition: pd.DataFrame, group_key: List[str]) -> pd.DataFrame:
    try:
        # Technically distributed is an optional dependency
        from distributed.protocol import serialize_bytes
    except ImportError:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return partition

    if partition.empty:
        res = partition[group_key]
        res[_PAYLOAD_COL] = b""
    else:
        res = partition.groupby(
            group_key,
            sort=False,
            observed=True,
            # Keep the as_index s.t. the group values are not dropped. With this
            # the behaviour seems to be consistent along pandas versions
            as_index=True,
        ).apply(lambda x: pd.Series({_PAYLOAD_COL: serialize_bytes(x)}))

        res = res.reset_index()
    return res


def pack_payload(df: dd.DataFrame, group_key: Union[List[str], str]) -> dd.DataFrame:
    """
    Pack all payload columns (everything except of group_key) into a single
    columns. This column will contain a single byte string containing the
    serialized and compressed payload data. The payload data is just dead weight
    when reshuffling. By compressing it once before the shuffle starts, this
    saves a lot of memory and network/disk IO.

    Example::

        >>> import pandas as pd
        ... import dask.dataframe as dd
        ... from dask.dataframe.shuffle import pack_payload
        ...
        ... df = pd.DataFrame({"A": [1, 1] * 2 + [2, 2] * 2 + [3, 3] * 2, "B": range(12)})
        ... ddf = dd.from_pandas(df, npartitions=2)

        >>> ddf.partitions[0].compute()

        A  B
        0  1  0
        1  1  1
        2  1  2
        3  1  3
        4  2  4
        5  2  5

        >>> pack_payload(ddf, "A").partitions[0].compute()

        A                               __dask_payload_bytes
        0  1  b'\x03\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x03...
        1  2  b'\x03\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x03...


    See also https://github.com/dask/dask/pull/6259

    """

    if (
        # https://github.com/pandas-dev/pandas/issues/34455
        isinstance(df._meta.index, pd.Float64Index)
        # TODO: Try to find out what's going on an file a bug report
        # For datetime indices the apply seems to be corrupt
        # s.t. apply(lambda x:x) returns different values
        or isinstance(df._meta.index, pd.DatetimeIndex)
    ):
        return df

    if not isinstance(group_key, list):
        group_key = [group_key]

    packed_meta = df._meta[group_key]
    packed_meta[_PAYLOAD_COL] = b""

    _pack_payload = partial(pack_payload_pandas, group_key=group_key)

    return df.map_partitions(_pack_payload, meta=packed_meta)


def unpack_payload_pandas(
    partition: pd.DataFrame, unpack_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Revert ``pack_payload_pandas`` and restore packed payload

    unpack_meta:
        A dataframe indicating the schema of the unpacked data. This will be returned in case the input is empty
    """
    try:
        # Technically distributed is an optional dependency
        from distributed.protocol import deserialize_bytes
    except ImportError:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return partition

    if partition.empty:
        return unpack_meta.iloc[:0]

    mapped = partition[_PAYLOAD_COL].map(deserialize_bytes)

    return pd.concat(mapped.values, copy=False, ignore_index=True)


def unpack_payload(df: dd.DataFrame, unpack_meta: pd.DataFrame) -> dd.DataFrame:
    """Revert payload packing of ``pack_payload`` and restores full dataframe."""

    if (
        # https://github.com/pandas-dev/pandas/issues/34455
        isinstance(df._meta.index, pd.Float64Index)
        # TODO: Try to find out what's going on an file a bug report
        # For datetime indices the apply seems to be corrupt
        # s.t. apply(lambda x:x) returns different values
        or isinstance(df._meta.index, pd.DatetimeIndex)
    ):
        return df

    return df.map_partitions(
        unpack_payload_pandas, unpack_meta=unpack_meta, meta=unpack_meta
    )
