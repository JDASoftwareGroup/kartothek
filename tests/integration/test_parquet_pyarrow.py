import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pandas.util.testing as pdtest
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import storefact
from pyarrow.parquet import ParquetFile

from kartothek.core._compat import ARROW_LARGER_EQ_0130
from kartothek.serialization import DataFrameSerializer, ParquetSerializer
from kartothek.serialization.testing import BINARY_COLUMNS, get_dataframe_not_nested


@pytest.fixture(params=BINARY_COLUMNS)
def binary_value(request):
    return request.param


@pytest.fixture
def dataframe_not_nested():
    return get_dataframe_not_nested(10)


@pytest.fixture
def reference_store():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "..",
        "reference-data",
        "pyarrow-bugs",
    )
    return storefact.get_store_from_url("hfs://{}".format(path))


def test_timestamp_us(store):
    # test that a df with us precision round-trips using parquet
    ts = datetime(2000, 1, 1, 15, 23, 24, 123456)
    df = pd.DataFrame({"ts": [ts]})
    serialiser = ParquetSerializer()
    key = serialiser.store(store, "prefix", df)
    pdtest.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


def test_pyarrow_07992(store):
    key = "test.parquet"
    df = pd.DataFrame({"a": [1]})
    table = pa.Table.from_pandas(df)
    meta = b"""{
        "pandas_version": "0.20.3",
        "index_columns": ["__index_level_0__"],
        "columns": [
            {"metadata": null, "name": "a", "numpy_type": "int64", "pandas_type": "int64"},
            {"metadata": null, "name": null, "numpy_type": "int64", "pandas_type": "int64"}
        ],
        "column_indexes": [
            {"metadata": null, "name": null, "numpy_type": "object", "pandas_type": "string"}
        ]
    }"""
    table = table.replace_schema_metadata({b"pandas": meta})
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    store.put(key, buf.getvalue().to_pybytes())
    pdtest.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


def test_index_metadata(store):
    key = "test.parquet"
    df = pd.DataFrame({"a": [1]})
    table = pa.Table.from_pandas(df)
    meta = b"""{
        "pandas_version": "0.20.3",
        "index_columns": ["__index_level_0__"],
        "columns": [
            {"metadata": null, "name": "a", "numpy_type": "int64", "pandas_type": "int64"}
        ]
    }"""
    table = table.replace_schema_metadata({b"pandas": meta})
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    store.put(key, buf.getvalue().to_pybytes())
    pdtest.assert_frame_equal(DataFrameSerializer.restore_dataframe(store, key), df)


@pytest.fixture(params=[1, 5, None])
def chunk_size(request, mocker):
    chunk_size = request.param
    if chunk_size == 1:
        # Test for a chunk size of one and mock the filter_df call. This way we can ensure that
        # the predicate for IO is properly evaluated and pushed down
        mocker.patch(
            "kartothek.serialization._parquet.filter_df",
            new=lambda x, *args, **kwargs: x,
        )
    return chunk_size


@pytest.mark.parametrize("use_categorical", [True, False])
def test_rowgroup_writing(store, use_categorical, chunk_size):
    df = pd.DataFrame({"string": ["abc", "affe", "banane", "buchstabe"]})
    serialiser = ParquetSerializer(chunk_size=2)
    # Arrow 0.9.0 has a bug in writing categorical columns to more than a single
    # RowGroup: "ArrowIOError: Column 2 had 2 while previous column had 4".
    # We have special handling for that in pandas-serialiser that should be
    # removed once we switch to 0.10.0
    if use_categorical:
        df_write = df.astype({"string": "category"})
    else:
        df_write = df
    key = serialiser.store(store, "prefix", df_write)

    parquet_file = ParquetFile(store.open(key))
    assert parquet_file.num_row_groups == 2


_INT_TYPES = ["int8", "int16", "int32", "int64"]

# uint64 will fail since numexpr cannot safe case from uint64 to int64
_UINT_TYPES = ["uint8", "uint16", "uint32"]

_FLOAT_TYPES = ["float32", "float64"]

_STR_TYPES = ["unicode", "bytes"]

_DATE_TYPES = ["date"]
_DATETIME_TYPES = ["datetime64"]


def _validate_predicate_pushdown(df, column, value, store, chunk_size):

    serialiser = ParquetSerializer(chunk_size=chunk_size)
    key = serialiser.store(store, "prefix", df)

    predicates = [[(column, "==", value)]]

    df_restored = serialiser.restore_dataframe(store, key, predicates=predicates)
    # date objects are converted to datetime in pyarrow
    df_restored["date"] = df_restored["date"].dt.date

    expected = df.iloc[[3]]
    # ARROW-5138 index isn't preserved when doing predicate pushdown
    if ARROW_LARGER_EQ_0130:
        pdt.assert_frame_equal(
            df_restored.reset_index(drop=True), expected.reset_index(drop=True)
        )
    else:
        pdt.assert_frame_equal(df_restored, expected)


@pytest.mark.parametrize("column", _STR_TYPES)
@pytest.mark.parametrize(
    "input_values", [(3, TypeError), (3.0, TypeError), ("3", None), (b"3", None)]
)
def test_predicate_evaluation_string(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _DATETIME_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        (datetime(2018, 1, 5), None),
        (np.datetime64(datetime(2018, 1, 5)), None),
        (pd.Timestamp(datetime(2018, 1, 5)), None),
        (np.datetime64(datetime(2018, 1, 5), "s"), None),
        (np.datetime64(datetime(2018, 1, 5), "ms"), None),
        (np.datetime64(datetime(2018, 1, 5), "us"), None),
        (np.datetime64(datetime(2018, 1, 5), "ns"), None),
        (date(2018, 1, 4), TypeError),
        ("2018-01-04", TypeError),
        (b"2018-01-04", TypeError),
        (1, TypeError),
        (1.0, TypeError),
    ],
)
def test_predicate_evaluation_datetime(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )
