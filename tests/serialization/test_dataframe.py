#!/usr/bin/env python
# -*- coding: utf-8 -*-


import datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

from kartothek.serialization import (
    CsvSerializer,
    DataFrameSerializer,
    ParquetSerializer,
    default_serializer,
)
from kartothek.serialization._util import ensure_unicode_string_type

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


def test_load_df_from_store_unsupported_format(store):
    with pytest.raises(ValueError):
        DataFrameSerializer.restore_dataframe(store, "test.unknown")


def test_store_df_to_store(store):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"]})
    dataframe_format = default_serializer()
    assert isinstance(dataframe_format, ParquetSerializer)
    key = dataframe_format.store(store, "prefix", df)
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_store_table_to_store(serialiser, store):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"]})
    table = pa.Table.from_pandas(df)
    key = serialiser.store(store, "prefix", table)
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_dataframe_roundtrip(serialiser, store):
    if serialiser in TYPE_STABLE_SERIALISERS:
        df = pd.DataFrame(
            {"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"], b"d": ["#", ";"]}
        )
        key = serialiser.store(store, "prefix", df)
        df.columns = [ensure_unicode_string_type(col) for col in df.columns]
    else:
        df = pd.DataFrame(
            {"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"], "d": ["#", ";"]}
        )
        key = serialiser.store(store, "prefix", df)

    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)

    # Test partial restore
    pdt.assert_frame_equal(
        DataFrameSerializer.restore_dataframe(store, key, columns=["a", "c"]),
        df[["a", "c"]],
    )

    # Test that all serialisers can ingest predicate_pushdown_to_io
    pdt.assert_frame_equal(
        DataFrameSerializer.restore_dataframe(
            store, key, columns=["a", "c"], predicate_pushdown_to_io=False
        ),
        df[["a", "c"]],
    )

    # Test that all serialisers can deal with categories
    expected = df[["c", "d"]].copy()
    expected["c"] = expected["c"].astype("category")
    # Check that the dtypes match but don't care about the order of the categoricals.
    pdt.assert_frame_equal(
        DataFrameSerializer.restore_dataframe(
            store, key, columns=["c", "d"], categories=["c"]
        ),
        expected,
        check_categorical=False,
    )

    # Test restore w/ empty col list
    pdt.assert_frame_equal(
        DataFrameSerializer.restore_dataframe(store, key, columns=[]), df[[]]
    )


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_missing_column(serialiser, store):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["∆", "€"], "d": ["#", ";"]})
    key = serialiser.store(store, "prefix", df)

    with pytest.raises(ValueError):
        DataFrameSerializer.restore_dataframe(store, key, columns=["a", "x"])


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_dataframe_roundtrip_empty(serialiser, store):
    df = pd.DataFrame({})
    key = serialiser.store(store, "prefix", df)
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)

    # Test partial restore
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


@pytest.mark.parametrize("serialiser", SERLIALISERS)
def test_dataframe_roundtrip_no_rows(serialiser, store):
    df = pd.DataFrame({"a": [], "b": [], "c": []}).astype(object)
    key = serialiser.store(store, "prefix", df)
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)

    # Test partial restore
    pdt.assert_frame_equal(
        DataFrameSerializer.restore_dataframe(store, key, columns=["a", "c"]),
        df[["a", "c"]],
    )


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
        (pd.DataFrame({"float_ü": [-3.141591, 0.0, 3.141593, 3.141595]}), {}),
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
    assert str(exc.value) == "Empty predicates"

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
    assert str(exc.value) == "Invalid predicates: Conjunction 0 is empty"

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
    assert str(exc.value) == "Invalid predicates: Conjunction 1 is empty"

    # Test malformed predicates 4
    predicates = [[(df.columns[0], "<", df.iloc[0, 0])], ["foo"]]
    with pytest.raises(ValueError) as exc:
        serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
            **read_kwargs,
        )
    assert (
        str(exc.value)
        == "Invalid predicates: Clause 0 in conjunction 1 should be a 3-tuple, got object of type <class 'str'> instead"
    )


@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_float_equal_big(predicate_pushdown_to_io, store, serialiser):
    df = pd.DataFrame({"float": [3141590.0, 3141592.0, 3141594.0]})
    key = serialiser.store(store, "prefix", df)

    predicates = [[("float", "==", 3141592.0)]]
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
    df = pd.DataFrame({"float": [0.3141590, 0.3141592, 0.3141594]})

    key = serialiser.store(store, "prefix", df)

    predicates = [[("float", "==", 0.3141592)]]
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


@type_stable_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_eval_string_types(serialiser, store, predicate_pushdown_to_io):
    df = pd.DataFrame({b"a": [1, 2], "b": [3.0, 4.0]})
    key = serialiser.store(store, "prefix", df)
    df.columns = [ensure_unicode_string_type(col) for col in df.columns]
    pdt.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)

    for col in ["a", b"a", "a"]:
        predicates = [[(col, "==", 1)]]
        result_df = serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
        )

        expected_df = df.iloc[[0], :].copy()

        pdt.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    for col in ["b", b"b", "b"]:
        predicates = [[(col, "==", 3.0)]]
        result_df = serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=predicates,
        )

        expected_df = df.iloc[[0], :].copy()

        pdt.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    for preds in (
        [[("a", "==", 1), ("b", "==", 3.0)]],
        [[("a", "==", 1), (b"b", "==", 3.0)]],
        [[(b"a", "==", 1), ("b", "==", 3.0)]],
    ):
        result_df = serialiser.restore_dataframe(
            store,
            key,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=preds,
        )

        expected_df = df.iloc[[0], :].copy()

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


@pytest.mark.parametrize(
    "df,value",
    [
        (pd.DataFrame({"nan": pd.Series([np.nan, -1.0, 1.0], dtype=float)}), 0.0),
        (pd.DataFrame({"inf": pd.Series([np.inf, -1.0, 1.0], dtype=float)}), 0.0),
        (pd.DataFrame({"ninf": pd.Series([-np.inf, -1.0, 1.0], dtype=float)}), 0.0),
        (
            pd.DataFrame(
                {"inf2": pd.Series([-np.inf, np.inf, -1.0, 1.0], dtype=float)}
            ),
            0.0,
        ),
        (
            pd.DataFrame(
                {"inf2": pd.Series([-np.inf, np.inf, -1.0, 1.0], dtype=float)}
            ),
            0.0,
        ),
        (
            pd.DataFrame(
                {"inf2": pd.Series([-np.inf, np.inf, -1.0, 1.0], dtype=float)}
            ),
            np.inf,
        ),
        (
            pd.DataFrame(
                {"inf2": pd.Series([-np.inf, np.inf, -1.0, 1.0], dtype=float)}
            ),
            -np.inf,
        ),
    ],
)
@predicate_serialisers
@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_predicate_pushdown_weird_floats_col(
    store, df, value, predicate_pushdown_to_io, serialiser
):
    key = serialiser.store(store, "prefix", df)

    col = df.columns[0]

    expected = df.loc[df[col] >= value].copy()
    predicates = [[(col, ">=", value)]]
    result = serialiser.restore_dataframe(
        store,
        key,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
    )
    assert_frame_almost_equal(result, expected)
