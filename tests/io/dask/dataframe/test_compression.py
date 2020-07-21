import dask.dataframe as dd
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.io.dask.compression import (
    pack_payload,
    pack_payload_pandas,
    unpack_payload,
    unpack_payload_pandas,
)


def test_pack_payload(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df = dd.from_pandas(
        pd.concat([df_all_types] * 10, ignore_index=True), npartitions=3
    )
    size_before = df.memory_usage(deep=True).sum()

    packed_df = pack_payload(df, group_key=list(df.columns[-2:]))

    size_after = packed_df.memory_usage(deep=True).sum()

    assert (size_after < size_before).compute()


def test_pack_payload_empty(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df_empty = dd.from_pandas(df_all_types.iloc[:0], npartitions=1)

    group_key = [df_all_types.columns[-1]]
    pdt.assert_frame_equal(
        df_empty.compute(),
        unpack_payload(
            pack_payload(df_empty, group_key=group_key), unpack_meta=df_empty._meta
        ).compute(),
    )


def test_pack_payload_pandas(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df = pd.concat([df_all_types] * 10, ignore_index=True)
    size_before = df.memory_usage(deep=True).sum()

    packed_df = pack_payload_pandas(df, group_key=list(df.columns[-2:]))

    size_after = packed_df.memory_usage(deep=True).sum()

    assert size_after < size_before


def test_pack_payload_pandas_empty(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df_empty = df_all_types.iloc[:0]

    group_key = [df_all_types.columns[-1]]
    pdt.assert_frame_equal(
        df_empty,
        unpack_payload_pandas(
            pack_payload_pandas(df_empty, group_key=group_key), unpack_meta=df_empty
        ),
    )


@pytest.mark.parametrize("num_group_cols", [1, 4])
def test_pack_payload_roundtrip(df_all_types, num_group_cols):
    group_key = list(df_all_types.columns[-num_group_cols:])
    df_all_types = dd.from_pandas(df_all_types, npartitions=2)
    pdt.assert_frame_equal(
        df_all_types.compute(),
        unpack_payload(
            pack_payload(df_all_types, group_key=group_key),
            unpack_meta=df_all_types._meta,
        ).compute(),
    )


@pytest.mark.parametrize("num_group_cols", [1, 4])
def test_pack_payload_pandas_roundtrip(df_all_types, num_group_cols):
    group_key = list(df_all_types.columns[-num_group_cols:])
    pdt.assert_frame_equal(
        df_all_types,
        unpack_payload_pandas(
            pack_payload_pandas(df_all_types, group_key=group_key),
            unpack_meta=df_all_types,
        ),
    )
