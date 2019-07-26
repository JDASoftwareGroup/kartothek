#!/usr/bin/env python
# -*- coding: utf-8 -*-


import datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pandas.util.testing as pdtest
import pyarrow as pa
import pytest

from kartothek.serialization import (
    CsvSerializer,
    DataFrameSerializer,
    ParquetSerializer,
    default_serializer,
)

TYPE_STABLE_SERIALISERS = [ParquetSerializer()]

SERLIALISERS = TYPE_STABLE_SERIALISERS + [
    CsvSerializer(),
    CsvSerializer(compress=False),
    default_serializer(),
]

type_stable_serialisers = pytest.mark.parametrize("serialiser", TYPE_STABLE_SERIALISERS)

predicate_serialisers = pytest.mark.parametrize(
    "serialiser",
    [
        ParquetSerializer(chunk_size=1),
        ParquetSerializer(chunk_size=2),
        ParquetSerializer(chunk_size=4),
    ]
    + SERLIALISERS,
)


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_store_table_to_store(serialiser, store):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"]})
    table = pa.Table.from_pandas(df)
    key = serialiser.store(store, "prefix", table)
    pdtest.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


def test_filter_query_predicate_exclusion(store):
    with pytest.raises(ValueError):
        DataFrameSerializer.restore_dataframe(
            store, "test.parquet", predicates=[[("a", "==", 1)]], filter_query="True"
        )


def assert_frame_almost_equal(df_left, df_right):
    """
    Be more friendly to some dtypes that are not preserved during the roundtrips.
    """
    # FIXME: This needs a better documentation
    for col in df_left.columns:
        if pd.api.types.is_datetime64_dtype(
            df_left[col].dtype
        ) and pd.api.types.is_object_dtype(df_right[col].dtype):
            df_right[col] = pd.to_datetime(df_right[col])
        elif pd.api.types.is_object_dtype(
            df_left[col].dtype
        ) and pd.api.types.is_datetime64_dtype(df_right[col].dtype):
            df_left[col] = pd.to_datetime(df_left[col])
        elif (
            len(df_left) > 0
            and pd.api.types.is_object_dtype(df_left[col].dtype)
            and pd.api.types.is_object_dtype(df_right[col].dtype)
        ):
            if isinstance(df_left[col].iloc[0], datetime.date) or isinstance(
                df_right[col].iloc[0], datetime.date
            ):
                df_left[col] = pd.to_datetime(df_left[col])
                df_right[col] = pd.to_datetime(df_right[col])
        elif pd.api.types.is_object_dtype(
            df_left[col].dtype
        ) and pd.api.types.is_categorical_dtype(df_right[col].dtype):
            df_left[col] = df_left[col].astype(df_right[col].dtype)
    pdt.assert_frame_equal(
        df_left.reset_index(drop=True), df_right.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "df, read_kwargs",
    [
        (pd.DataFrame({"string_ü": ["abc", "affe", "banane", "buchstabe_ü"]}), {}),
        (pd.DataFrame({"integer_ü": np.arange(4)}), {}),
        (pd.DataFrame({"float_ü": [-3.141_591, 0.0, 3.141_593, 3.141_595]}), {}),
        (
            pd.DataFrame(
                {
                    "date_ü": [
                        datetime.date(2011, 1, 31),
                        datetime.date(2011, 2, 3),
                        datetime.date(2011, 2, 4),
                        datetime.date(2011, 3, 10),
                    ]
                }
            ),
            {"date_as_object": False},
        ),
        (
            pd.DataFrame(
                {
                    "date_ü": [
                        datetime.date(2011, 1, 31),
                        datetime.date(2011, 2, 3),
                        datetime.date(2011, 2, 4),
                        datetime.date(2011, 3, 10),
                    ]
                }
            ),
            {"date_as_object": True},
        ),
        (
            pd.DataFrame(
                {"categorical_ü": list("abcd")},
                dtype=pd.api.types.CategoricalDtype(list("abcd"), ordered=True),
            ),
            {},
        ),
    ],
)
@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_pushdown(
    store, df, read_kwargs, predicate_pushdown_to_io, serialiser
):
    """
    Test predicate pushdown for several types and operations.

    The DataFrame parameters all need to be of same length for this test to
    work universally. Also the values in the DataFrames need to be sorted in
    ascending order.
    """
    # All test dataframes need to have the same length
    assert len(df) == 4
    assert df[df.columns[0]].is_monotonic and df.iloc[0, 0] < df.iloc[-1, 0]

    # This is due to the limitation that dates cannot be expressed in
    # Pandas' query() method.
    if isinstance(serialiser, CsvSerializer) and isinstance(
        df.iloc[0, 0], datetime.date
    ):
        pytest.skip("CsvSerialiser cannot filter on dates")

    key = serialiser.store(store, "prefix", df)

    # Test `<` and `>` operators
    expected = df.iloc[[1, 2], :].copy()
    predicates = [
        [(df.columns[0], "<", df.iloc[3, 0]), (df.columns[0], ">", df.iloc[0, 0])]
    ]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test `=<` and `>=` operators
    expected = df.iloc[[1, 2, 3], :].copy()
    predicates = [
        [(df.columns[0], "<=", df.iloc[3, 0]), (df.columns[0], ">=", df.iloc[1, 0])]
    ]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test `==` operator
    expected = df.iloc[[1], :].copy()
    predicates = [[(df.columns[0], "==", df.iloc[1, 0])]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test `in` operator
    expected = df.iloc[[1], :].copy()
    predicates = [[(df.columns[0], "in", [df.iloc[1, 0]])]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test `!=` operator
    expected = df.iloc[[0, 2, 3], :].copy()
    predicates = [[(df.columns[0], "!=", df.iloc[1, 0])]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test empty DataFrame
    expected = df.head(0)
    predicates = [[(df.columns[0], "<", df.iloc[0, 0])]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test in empty list
    expected = df.head(0)
    predicates = [[(df.columns[0], "in", [])]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test in numpy array
    expected = df.iloc[[1], :].copy()
    predicates = [[(df.columns[0], "in", np.asarray([df.iloc[1, 0], df.iloc[1, 0]]))]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
        **read_kwargs,
    )
    assert_frame_almost_equal(result, expected)

    # Test malformed predicates 1
    predicates = []
    with pytest.raises(ValueError) as exc:
        serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
            **read_kwargs,
        )
    assert str(exc.value) == "Malformed predicates"

    # Test malformed predicates 2
    predicates = [[]]
    with pytest.raises(ValueError) as exc:
        serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
            **read_kwargs,
        )
    assert str(exc.value) == "Malformed predicates"

    # Test malformed predicates 3
    predicates = [[(df.columns[0], "<", df.iloc[0, 0])], []]
    with pytest.raises(ValueError) as exc:
        serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
            **read_kwargs,
        )
    assert str(exc.value) == "Malformed predicates"


@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_float_equal_big(predicate_pushdown_to_io, store, serialiser):
    df = pd.DataFrame({"float": [3_141_590.0, 3_141_592.0, 3_141_594.0]})
    key = serialiser.store(store, "prefix", df)

    predicates = [[("float", "==", 3_141_592.0)]]
    result_df = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
    )
    expected_df = df.iloc[[1], :].copy()

    pdt.assert_frame_equal(
        result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )


@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_float_equal_small(predicate_pushdown_to_io, store, serialiser):
    df = pd.DataFrame({"float": [0.314_159_0, 0.314_159_2, 0.314_159_4]})

    key = serialiser.store(store, "prefix", df)

    predicates = [[("float", "==", 0.314_159_2)]]
    result_df = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
    )
    expected_df = df.iloc[[1], :].copy()

    pdt.assert_frame_equal(
        result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "df,value",
    [
        (pd.DataFrame({"u": pd.Series([None], dtype=object)}), "foo"),
        (pd.DataFrame({"b": pd.Series([None], dtype=object)}), b"foo"),
        (pd.DataFrame({"f": pd.Series([np.nan], dtype=float)}), 1.2),
        (
            pd.DataFrame({"t": pd.Series([pd.NaT], dtype="datetime64[ns]")}),
            pd.Timestamp("2017"),
        ),
    ],
)
@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_pushdown_null_col(
    store, df, value, predicate_pushdown_to_io, serialiser
):
    key = serialiser.store(store, "prefix", df)

    expected = df.iloc[[]].copy()
    predicates = [[(df.columns[0], "==", value)]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=serialiser.type_stable,
    )
