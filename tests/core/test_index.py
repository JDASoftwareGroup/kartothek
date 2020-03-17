# -*- coding: utf-8 -*-


import datetime
import logging
import pickle
from itertools import permutations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import pytz
from hypothesis import assume, given
from pandas.testing import assert_series_equal

from kartothek.core._compat import ARROW_LARGER_EQ_0150
from kartothek.core.index import ExplicitSecondaryIndex, IndexBase, merge_indices
from kartothek.core.testing import get_numpy_array_strategy


@pytest.mark.parametrize("inplace", [True, False])
def test_index_update(inplace):
    original_index = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_1", "part_2"], 3: ["part_3"]}
    )

    new_index = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_4"], 4: ["part_4"]}
    )

    updated_index = original_index.update(new_index, inplace=inplace)

    expected_index = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_2", "part_4", "part_1"], 3: ["part_3"], 4: ["part_4"]},
    )
    assert updated_index == expected_index


@pytest.mark.parametrize("inplace", [True, False])
def test_storage_key_after_update(inplace):
    """
    Assert that the storage key is not set after mutation of the index object
    """
    original_index = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 3: ["part_3"]},
        index_storage_key="storage_key",
    )
    updated_index = original_index.remove_partitions([], inplace=inplace)
    assert updated_index.index_storage_key == "storage_key"
    updated_index = original_index.remove_partitions(["part_1"], inplace=inplace)
    assert updated_index.index_storage_key is None

    original_index = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 3: ["part_3"]},
        index_storage_key="storage_key",
    )
    updated_index = original_index.remove_values([], inplace=inplace)
    assert updated_index.index_storage_key == "storage_key"

    updated_index = original_index.remove_values([1], inplace=inplace)
    assert updated_index.index_storage_key is None

    original_index = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 3: ["part_3"]},
        index_storage_key="storage_key",
    )
    updated_index = original_index.copy()
    assert updated_index.index_storage_key == "storage_key"
    updated_index = original_index.copy(column="something_different")
    assert updated_index.index_storage_key is None


def test_eq_explicit():
    def assert_eq(a, b):
        assert a == b
        assert b == a
        assert not (a != b)
        assert not (b != a)

    def assert_ne(a, b):
        assert a != b
        assert b != a
        assert not (a == b)
        assert not (b == a)

    original_index = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1"]},
        dtype=pa.int64(),
        index_storage_key="dataset_uuid/some_index.parquet",
    )

    idx1 = original_index.copy()
    assert_eq(idx1, original_index)

    idx2 = original_index.copy()
    idx2.column = "col2"
    assert_ne(idx2, original_index)

    idx3 = original_index.copy()
    idx3.dtype = pa.uint64()
    assert_ne(idx3, original_index)

    idx4 = original_index.copy()
    idx4.index_dct = {1: ["part_1"], 2: ["part_2"]}
    assert_ne(idx4, original_index)

    idx5 = original_index.copy()
    idx5.index_dct = {1: ["part_1", "part_2"]}
    assert_ne(idx5, original_index)

    idx6 = original_index.copy()
    idx6.index_dct = {1: ["part_2"]}
    assert_ne(idx6, original_index)

    idx7 = original_index.copy()
    idx7.index_dct = {2: ["part_1"]}
    assert_ne(idx7, original_index)

    idx8 = original_index.copy()
    idx8.dtype = None
    assert_ne(idx8, original_index)

    idx9a = original_index.copy()
    idx9b = original_index.copy()
    idx9a.dtype = None
    idx9b.dtype = None
    assert_eq(idx9a, idx9b)


def test_index_update_wrong_col():
    original_index = ExplicitSecondaryIndex(column="col", index_dct={1: ["part_1"]})

    new_index = ExplicitSecondaryIndex(column="another_col", index_dct={1: ["part_4"]})
    with pytest.raises(ValueError) as e:
        original_index.update(new_index)
    assert (
        str(e.value)
        == "Trying to update an index with the wrong column. Got `another_col` but expected `col`"
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pa.binary(),
        pa.bool_(),
        pa.date32(),
        pa.float32(),
        pa.float64(),
        pa.int64(),
        pa.int8(),
        pa.string(),
        pa.timestamp("ns"),
    ],
)
def test_index_empty(store, dtype):
    storage_key = "dataset_uuid/some_index.parquet"
    index1 = ExplicitSecondaryIndex(
        column="col", index_dct={}, dtype=dtype, index_storage_key=storage_key
    )
    key1 = index1.store(store, "dataset_uuid")

    index2 = ExplicitSecondaryIndex(column="col", index_storage_key=key1).load(store)
    assert index1 == index2

    index3 = pickle.loads(pickle.dumps(index1))
    assert index1 == index3


def test_pickle_without_load(store):
    storage_key = "dataset_uuid/some_index.parquet"
    index1 = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_1"]}, index_storage_key=storage_key
    )
    key1 = index1.store(store, "dataset_uuid")

    index2 = ExplicitSecondaryIndex(column="col", index_storage_key=key1)
    assert index2 != index1

    index3 = pickle.loads(pickle.dumps(index2))
    assert index3 == index2

    index4 = index3.load(store)
    assert index4 == index1
    assert index4 != index2


def test_index_no_source():
    with pytest.raises(ValueError) as e:
        ExplicitSecondaryIndex(column="col")
    assert str(e.value) == "No valid index source specified"


@pytest.mark.parametrize("inplace", [True, False])
def test_index_remove_values(inplace):
    original_index = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_1", "part_2"], 2: ["part_1"], 3: ["part_3"]}
    )
    new_index = original_index.remove_values([1, 2], inplace=inplace)
    expected_index = ExplicitSecondaryIndex(column="col", index_dct={3: ["part_3"]})
    assert new_index == expected_index


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "remove, expected",
    [
        (["part_1"], {1: ["part_2"], 3: ["part_3"]}),
        (["part_1", "part_2"], {3: ["part_3"]}),
        (["part_1", "part_2", "part_3"], {}),
    ],
)
def test_index_remove_partitions(inplace, remove, expected):
    original_index = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_1", "part_2"], 2: ["part_1"], 3: ["part_3"]}
    )
    for perm in permutations(remove):
        new_index = original_index.remove_partitions(perm, inplace=inplace)
        expected_index = ExplicitSecondaryIndex(
            column="col", index_dct=expected, dtype=pa.int64()
        )
        assert new_index == expected_index


def test_index_store_roundtrip_explicit_key(store):
    storage_key = "dataset_uuid/some_index.parquet"
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 3: ["part_3"]},
        index_storage_key=storage_key,
        dtype=pa.int64(),
    )
    key1 = index1.store(store, "dataset_uuid")

    index2 = ExplicitSecondaryIndex(column="col", index_storage_key=key1).load(store)
    assert index1 == index2
    key2 = index2.store(store, "dataset_uuid")

    index3 = ExplicitSecondaryIndex(column="col", index_storage_key=key2).load(store)
    assert index1 == index3
    assert index2 == index3


@pytest.mark.parametrize("col", ["col", "foo/bar", "foo:bar"])
def test_index_store_roundtrip_implicit_key(store, col):
    index1 = ExplicitSecondaryIndex(
        column=col, index_dct={1: ["part_1", "part_2"], 3: ["part_3"]}, dtype=pa.int64()
    )
    key1 = index1.store(store, "dataset_uuid")
    index1.index_storage_key = key1

    index2 = ExplicitSecondaryIndex(column=col, index_storage_key=key1).load(store)
    assert index1 == index2
    key2 = index2.store(store, "dataset_uuid")

    index3 = ExplicitSecondaryIndex(column=col, index_storage_key=key2).load(store)
    assert index1 == index3
    assert index2 == index3


def test_index_as_flat_series():
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 2: ["part_1"]},
        dtype=pa.int64(),
    )
    ser = index1.as_flat_series()
    expected = pd.Series(
        ["part_1", "part_2", "part_1"],
        index=pd.Index([1, 1, 2], name="col"),
        name="partition",
    )
    assert_series_equal(ser, expected)

    ser_comp = index1.as_flat_series(compact=True)
    expected = pd.Series(
        [["part_1", "part_2"], ["part_1"]],
        index=pd.Index([1, 2], name="col"),
        name="partition",
    )
    assert_series_equal(ser_comp, expected)


def test_index_as_flat_series_single_value():

    index1 = ExplicitSecondaryIndex(
        column="col", index_dct={1: ["part_1", "part_2"]}, dtype=pa.int64()
    )
    ser = index1.as_flat_series()
    expected = pd.Series(
        ["part_1", "part_2"], index=pd.Index([1, 1], name="col"), name="partition"
    )
    assert_series_equal(ser, expected)

    ser_comp = index1.as_flat_series(compact=True)
    expected = pd.Series(
        [["part_1", "part_2"]], index=pd.Index([1], name="col"), name="partition"
    )
    assert_series_equal(ser_comp, expected)


def test_index_as_flat_series_partitions_as_index():

    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1", "part_2"], 2: ["part_1"]},
        dtype=pa.int64(),
    )

    ser = index1.as_flat_series(partitions_as_index=True)
    expected = pd.Series(
        [1, 1, 2],
        index=pd.Index(["part_1", "part_2", "part_1"], name="partition"),
        name="col",
    )
    assert_series_equal(ser, expected)

    ser_comp = index1.as_flat_series(compact=True, partitions_as_index=True)
    expected = pd.Series(
        [[1, 2], [1]],
        index=pd.Index(["part_1", "part_2"], name="partition"),
        name="col",
    )
    assert_series_equal(ser_comp, expected)


def test_index_as_flat_series_highly_degenerated_sym():
    dim = 4
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={
            k: ["part_{}".format(i) for i in range(0, dim)] for k in range(0, dim)
        },
        dtype=pa.int64(),
    )
    ser = index1.as_flat_series()
    expected = pd.Series(
        ["part_{}".format(i) for i in range(0, dim)] * dim,
        index=pd.Index(
            np.array([[i] * dim for i in range(0, dim)]).ravel(), name="col"
        ),
        name="partition",
    )
    assert_series_equal(ser, expected)


def test_index_as_flat_series_highly_degenerated_asym():
    """
    Ensure that the generation of the series is not bound by col numbers or nans in the matrix
    """
    dim = 4
    ind_dct = {k: ["part_{}".format(i) for i in range(0, dim)] for k in range(0, dim)}
    ind_dct[0] = ["part_1"]
    ind_dct[2] = ["part_2", "part_5"]
    index1 = ExplicitSecondaryIndex(column="col", index_dct=ind_dct, dtype=pa.int64())
    ser = index1.as_flat_series()
    partition = [
        "part_1",
        "part_0",
        "part_1",
        "part_2",
        "part_3",
        "part_2",
        "part_5",
        "part_0",
        "part_1",
        "part_2",
        "part_3",
    ]
    index_values = [0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    expected = pd.Series(
        partition, index=pd.Index(index_values, name="col", dtype=int), name="partition"
    )
    assert_series_equal(ser, expected)

    ser_inv = index1.as_flat_series(partitions_as_index=True)
    expected_inv = pd.Series(
        index_values, index=pd.Index(partition, name="partition"), name="col"
    )
    assert_series_equal(ser_inv, expected_inv)


@pytest.mark.parametrize(
    "dtype, date_as_object", [(None, True), ("datetime64[ns]", False)]
)
def test_index_as_flat_series_date(dtype, date_as_object):
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={
            datetime.date(2017, 1, 2): ["part_1", "part_2"],
            datetime.date(2018, 2, 3): ["part_1"],
        },
        dtype=pa.date32(),
    )
    ser = index1.as_flat_series(date_as_object=date_as_object)
    ser = ser.sort_index()
    expected = pd.Series(
        ["part_1", "part_2", "part_1"],
        index=pd.Index(
            [
                datetime.date(2017, 1, 2),
                datetime.date(2017, 1, 2),
                datetime.date(2018, 2, 3),
            ],
            dtype=dtype,
            name="col",
        ),
        name="partition",
    )
    assert_series_equal(ser, expected)


@pytest.mark.parametrize(
    "dtype, timestamps",
    [
        (pa.timestamp("ns"), [pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02")]),
        (
            pa.timestamp("ns"),
            [
                pd.Timestamp("2017-01-01", tzinfo=pytz.timezone("EST")),
                pd.Timestamp("2017-01-02", tzinfo=pytz.timezone("EST")),
            ],
        ),
    ],
)
def test_index_store_roundtrip_ts(store, dtype, timestamps):
    storage_key = "dataset_uuid/some_index.parquet"
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct=dict(zip(timestamps, [["part_1", "part_2"], ["part_3"]])),
        index_storage_key=storage_key,
        dtype=dtype,
    )
    key1 = index1.store(store, "dataset_uuid")

    index2 = ExplicitSecondaryIndex(column="col", index_storage_key=key1).load(store)
    assert index1 == index2


@pytest.mark.parametrize(
    "dtype,expected", [(pa.int8(), pa.int64()), (pa.uint8(), pa.uint64()), (None, None)]
)
def test_index_normalize_dtype(dtype, expected):
    index = ExplicitSecondaryIndex(
        column="col", dtype=dtype, index_storage_key="dataset_uuid/some_index.parquet"
    )
    assert index.dtype == expected


def test_index_raises_nested_dtype():
    with pytest.raises(NotImplementedError) as exc:
        ExplicitSecondaryIndex(
            column="col",
            dtype=pa.list_(pa.int8()),
            index_storage_key="dataset_uuid/some_index.parquet",
        )
    assert str(exc.value) == "Indices w/ nested types are not supported"


def test_index_raises_null_dtype():
    with pytest.raises(NotImplementedError) as exc:
        ExplicitSecondaryIndex(
            column="col",
            dtype=pa.null(),
            index_storage_key="dataset_uuid/some_index.parquet",
        )
    assert str(exc.value) == "Indices w/ null/NA type are not supported"


@pytest.mark.parametrize(
    "dtype,value",
    [
        (pa.bool_(), True),
        (pa.int64(), 1),
        (pa.float64(), 1.1),
        (pa.binary(), b"x"),
        (pa.string(), "x"),
        (pa.timestamp("ns"), pd.Timestamp("2018-01-01").to_datetime64()),
        (pa.date32(), datetime.date(2018, 1, 1)),
        pytest.param(
            pa.timestamp("ns", tz=pytz.timezone("Europe/Berlin")),
            pd.Timestamp("2018-01-01", tzinfo=pytz.timezone("Europe/Berlin")),
            marks=pytest.mark.xfail(
                not ARROW_LARGER_EQ_0150,
                reason="Timezone reoundtrips not supported in older versions",
            ),
        ),
    ],
)
def test_observed_values_plain(dtype, value):
    ind = ExplicitSecondaryIndex(
        column="col", dtype=dtype, index_dct={value: ["part_label"]}
    )
    observed = ind.observed_values()
    assert len(observed) == 1
    assert list(observed) == [value]


@pytest.mark.parametrize("date_as_object", [None, True, False])
def test_observed_values_date_as_object(date_as_object):
    value = datetime.date(2020, 1, 1)
    ind = ExplicitSecondaryIndex(
        column="col", dtype=pa.date32(), index_dct={value: ["part_label"]}
    )
    observed = ind.observed_values(date_as_object=date_as_object)
    if date_as_object:
        expected = value
    else:
        expected = pd.Timestamp(value).to_datetime64()
    assert len(observed) == 1
    assert observed[0] == expected


@pytest.mark.parametrize(
    "dtype,value,expected",
    [
        (pa.bool_(), True, True),
        (pa.bool_(), False, False),
        (pa.bool_(), 1, True),
        (pa.bool_(), 0, False),
        (pa.bool_(), "True", True),
        (pa.bool_(), "False", False),
        (pa.bool_(), "true", True),
        (pa.bool_(), "false", False),
        (pa.int64(), 1, 1),
        (pa.int64(), "1", 1),
        (pa.int64(), 1.0, 1),
        (pa.float64(), 1.1, 1.1),
        (pa.float64(), "1.1", 1.1),
        (pa.float64(), 1, 1.0),
        (pa.binary(), "x", b"x"),
        (pa.string(), "x", "x"),
        (pa.string(), "ö", "ö"),
        (pa.string(), 1, "1"),
        (pa.string(), "ö".encode("utf8"), "ö"),
        (
            pa.timestamp("ns"),
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-01-01").to_datetime64(),
        ),
        (
            pa.timestamp("ns"),
            pd.Timestamp("2018-01-01").to_datetime64(),
            pd.Timestamp("2018-01-01").to_datetime64(),
        ),
        (pa.timestamp("ns"), "2018-01-01", pd.Timestamp("2018-01-01").to_datetime64()),
        (pa.date32(), "2018-01-01", datetime.date(2018, 1, 1)),
        (
            pa.timestamp("ns", tz=pytz.timezone("Europe/Berlin")),
            pd.Timestamp("2018-01-01", tzinfo=pytz.timezone("Europe/Berlin")),
            pd.Timestamp(
                "2018-01-01", tzinfo=pytz.timezone("Europe/Berlin")
            ).to_datetime64(),
        ),
        (
            pa.timestamp("ns", tz=pytz.timezone("Europe/Berlin")),
            "2018-01-01",  # Naive date, is interpreted as being UTC
            pd.Timestamp("2018-01-01", tzinfo=pytz.timezone("UTC")).to_datetime64(),
        ),
    ],
)
def test_index_normalize_value(dtype, value, expected):
    index = ExplicitSecondaryIndex(
        column="col", dtype=dtype, index_storage_key="dataset_uuid/some_index.parquet"
    )
    actual = index.normalize_value(index.dtype, value)
    assert actual == expected
    assert type(actual) == type(expected)


def test_index_normalize_during_init():
    index = ExplicitSecondaryIndex(
        column="col",
        dtype=pa.int8(),
        index_dct={"1": ["a", "b"], 1: ["a", "c"], 2.0: ["d"]},
    )
    expected = {1: ["a", "b", "c"], 2: ["d"]}
    assert index.index_dct == expected


@pytest.mark.parametrize("collision", [True, False])
def test_index_normalize_during_init_warn_collision(collision, caplog):
    index_dct = {1: ["a", "c"], 2.0: ["d"]}
    if collision:
        index_dct["1"] = ["a", "b"]

    caplog.set_level(logging.DEBUG)
    ExplicitSecondaryIndex(column="col", dtype=pa.int8(), index_dct=index_dct)

    warn = [
        t[2]
        for t in caplog.record_tuples
        if t[0] == "kartothek.core.index" and t[1] == logging.WARN
    ]

    if collision:
        assert any(
            msg.startswith(
                "Value normalization for index column col resulted in 1 collision(s)."
            )
            for msg in warn
        )
    else:
        assert not any(
            msg.startswith("Value normalization for index column") for msg in warn
        )


def test_index_normalize_during_query():
    index = ExplicitSecondaryIndex(
        column="col", dtype=pa.int64(), index_dct={1: ["a", "b", "c"], 2: ["d"]}
    )
    assert index.query(1) == ["a", "b", "c"]
    assert index.query(2) == ["d"]
    assert index.query("2") == ["d"]
    assert index.query(1.0) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "op, value, expected",
    [
        ("==", 1, {"b", "c", "e"}),
        ("<=", 1, {"a", "b", "c", "e"}),
        (">=", 1, {"b", "c", "e", "f"}),
        ("<", 1, {"a", "b", "c"}),
        (">", 1, {"f"}),
        ("in", [0, 2], {"a", "b", "c", "f"}),
    ],
)
@given(index_data=get_numpy_array_strategy(unique=True, sort=True, allow_nan=False))
def test_eval_operators(index_data, op, value, expected):
    index = ExplicitSecondaryIndex(
        column="col",
        index_dct={
            index_data[0]: ["a", "b", "c"],
            index_data[1]: ["b", "c", "e"],
            index_data[2]: ["f"],
        },
    )
    assume(len(index.index_dct) == 3)
    result = index.eval_operator(op, index_data[value])

    assert result == expected


def test_eval_operators_type_safety():
    # gh66
    ind = IndexBase(column="col", index_dct={1234: ["part"]}, dtype=pa.int64())
    with pytest.raises(
        TypeError,
        match=r"Unexpected type for predicate: Column 'col' has pandas type 'int64', "
        r"but predicate value '1234' has pandas type 'object' \(Python type '<class 'str'>'\).",
    ):
        ind.eval_operator("==", "1234")
    with pytest.raises(
        TypeError,
        match=r"Unexpected type for predicate: Column 'col' has pandas type 'int64', "
        r"but predicate value 1234.0 has pandas type 'float64' \(Python type '<class 'float'>'\).",
    ):
        ind.eval_operator("==", 1234.0)

    assert ind.eval_operator("==", 1234) == {"part"}


@pytest.mark.parametrize("inplace", [True, False])
def test_index_normalize_remove_values(inplace):
    original_index = ExplicitSecondaryIndex(
        column="col", dtype=pa.int64(), index_dct={1: ["a", "b", "c"], 2: ["d"]}
    )

    new_index1 = original_index.copy().remove_values([1, 3], inplace=inplace)
    expected_index1 = ExplicitSecondaryIndex(
        column="col", dtype=pa.int64(), index_dct={2: ["d"]}
    )
    assert new_index1 == expected_index1

    new_index2 = original_index.copy().remove_values([1.0, 3.0], inplace=inplace)
    expected_index2 = ExplicitSecondaryIndex(
        column="col", dtype=pa.int64(), index_dct={2: ["d"]}
    )
    assert new_index2 == expected_index2

    new_index3 = original_index.copy().remove_values(["1", "3"], inplace=inplace)
    expected_index3 = ExplicitSecondaryIndex(
        column="col", dtype=pa.int64(), index_dct={2: ["d"]}
    )
    assert new_index3 == expected_index3


def test_index_ts_inference(store):
    index = ExplicitSecondaryIndex(
        column="col",
        index_dct={
            pd.Timestamp("2017-01-01"): ["part_1", "part_2"],
            pd.Timestamp("2017-01-02"): ["part_3"],
        },
    )
    assert index.dtype == pa.timestamp("ns")


def _dict_to_index(dct):
    new_dct = {}
    for col in dct:
        new_dct[col] = ExplicitSecondaryIndex(col, dct[col])
    return new_dct


@pytest.mark.parametrize("obj_factory", [dict, _dict_to_index])
def test_merge_indices(obj_factory):
    indices = [
        _dict_to_index({"location": {"Loc1": ["label1"], "Loc2": ["label1"]}}),
        _dict_to_index(
            {
                "location": {"Loc3": ["label2"], "Loc2": ["label2"]},
                "product": {"Product1": ["label2"], "Product2": ["label2"]},
            }
        ),
        _dict_to_index(
            {"location": {"Loc4": ["label3"]}, "product": {"Product1": ["label3"]}}
        ),
    ]
    result = merge_indices(indices)
    for key, value in result.items():
        if isinstance(value, ExplicitSecondaryIndex):
            result[key] = value.to_dict()
            for part_list in result[key].values():
                part_list.sort()

    expected = {
        "location": {
            "Loc1": ["label1"],
            "Loc2": ["label1", "label2"],
            "Loc3": ["label2"],
            "Loc4": ["label3"],
        },
        "product": {"Product1": ["label2", "label3"], "Product2": ["label2"]},
    }
    assert result == expected


def test_index_uint():
    index = ExplicitSecondaryIndex(
        column="col",
        index_dct={
            14671423800646041619: ["part_1", "part_2"],
            np.iinfo(np.uint64).max: ["part_1"],
        },
    )
    assert index.dtype == "uint64"


@pytest.mark.parametrize(
    "key",
    [
        True,  # pa.bool_()
        1,  # pa.int64()
        1.1,  # pa.float64()
        b"x",  # pa.binary()
        "ö",  # pa.string()
        pd.Timestamp("2018-01-01").to_datetime64(),  # pa.timestamp("ns")
        pd.Timestamp(
            "2018-01-01", tzinfo=pytz.timezone("Europe/Berlin")
        ).to_datetime64(),  # pa.timestamp("ns")
        datetime.date(2018, 1, 1),  # pa.date32()
    ],
)
def test_serialization(key):
    """Check index remains consistent after serializing and de-serializing"""
    index = ExplicitSecondaryIndex(
        column="col", index_dct={key: ["part_2", "part_4", "part_1"]}
    )
    index2 = pickle.loads(pickle.dumps(index))

    assert index == index2


@pytest.mark.parametrize(
    "key",
    [
        True,  # pa.bool_()
        1,  # pa.int64()
        1.1,  # pa.float64()
        b"x",  # pa.binary()
        "ö",  # pa.string()
        pd.Timestamp("2018-01-01").to_datetime64(),  # pa.timestamp("ns")
        pd.Timestamp(
            "2018-01-01", tzinfo=pytz.timezone("Europe/Berlin")
        ).to_datetime64(),  # pa.timestamp("ns")
        datetime.datetime(
            2018, 1, 1, 12, 30
        ),  # pa.timestamp("us") (initial) => pa.timestamp("ns") (after loading)
        datetime.datetime(
            2018, 1, 1, 12, 30, tzinfo=pytz.timezone("Europe/Berlin")
        ),  # pa.timestamp("ns")
        datetime.date(2018, 1, 1),  # pa.date32()
    ],
)
def test_serialization_normalization(key):
    """
    Check that index normalizes values consistently after serializing.

    This is helpful to ensure correct behavior for cases such as when
    key=`datetime.datetime(2018, 1, 1, 12, 30)`, as this would be parsed to
    `pa.timestamp("us")` during index creation, but stored as `pa.timestamp("ns")`.
    """
    index = ExplicitSecondaryIndex(
        column="col", index_dct={key: ["part_2", "part_4", "part_1"]}
    )
    index2 = pickle.loads(pickle.dumps(index))

    assert index.normalize_value(index.dtype, key) == index2.normalize_value(
        index2.dtype, key
    )


@pytest.mark.parametrize("with_index_dct", [True, False])
def test_unload(with_index_dct):
    storage_key = "dataset_uuid/some_index.parquet"
    index1 = ExplicitSecondaryIndex(
        column="col",
        index_dct={1: ["part_1"]} if with_index_dct else None,
        index_storage_key=storage_key,
    )

    index2 = index1.unload()
    assert not index2.loaded
    assert index2.index_storage_key == storage_key

    index3 = pickle.loads(pickle.dumps(index2))
    assert index2 == index3


def test_fail_type_unsafe():
    with pytest.raises(ValueError, match="Trying to create non-typesafe index"):
        ExplicitSecondaryIndex(column="col", index_dct={})
