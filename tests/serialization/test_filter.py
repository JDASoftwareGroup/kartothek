import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import testing as pdt

from kartothek.serialization import filter_array_like, filter_df_from_predicates


@pytest.fixture(
    params=[
        np.array(range(10)),
        np.array(range(10)).astype(float),
        pd.Series(np.array(range(10))),
        pd.Series(np.array(range(10)).astype(float)),
        pd.Series(np.array([str(x) for x in range(10)])),
        pd.Series(np.array([str(x) for x in range(10)])).astype("category"),
        np.array(["A", "B", "C", "D", "E", "F"], dtype="<U1"),
        np.array(["A", "B", "C", "D", "E", "F"], dtype="S"),
    ]
)
def array_like(request):
    return request.param


def test_filter_array_like_eq(array_like):
    ix = 3
    value = array_like[ix]
    res = filter_array_like(array_like, "==", value)

    assert all(array_like[res] == array_like[ix])


def test_filter_array_like_larger_eq(array_like):
    ix = 3
    value = array_like[ix]
    res = filter_array_like(array_like, ">=", value)

    assert all(array_like[res] == array_like[ix:])


def test_filter_array_like_lower_eq(array_like):
    ix = 3
    value = array_like[ix]
    res = filter_array_like(array_like, "<=", value)

    assert all(array_like[res] == array_like[: ix + 1])


@pytest.mark.parametrize(
    "op, expected",
    [
        (">", [False, True, False]),
        (">=", [True, True, False]),
        ("<=", [True, False, True]),
        ("<", [False, False, True]),
    ],
)
@pytest.mark.parametrize(
    "cat_type",
    [
        "category",
        pd.CategoricalDtype(["A", "B", "C"], ordered=False),
        pd.CategoricalDtype(["B", "C", "A"], ordered=False),
        pd.CategoricalDtype(["A", "B", "C"], ordered=True),
    ],
)
def test_filter_array_like_categoricals(op, expected, cat_type):
    ser = pd.Series(["B", "C", "A"]).astype(cat_type)
    res = filter_array_like(ser, op, "B")

    assert all(res == expected)


@pytest.mark.parametrize(
    "value, filter_value",
    [
        (1, 1.0),
        ("1", 1.0),
        ("1", 1),
        (1, "1"),
        (datetime.date(2019, 1, 1), 1),
        (datetime.datetime(2019, 1, 1), 1),
        (datetime.datetime(2019, 1, 1), datetime.date(2019, 1, 1)),
        (datetime.date(2019, 1, 1), datetime.datetime(2019, 1, 1)),
        (True, 1),
        (True, 1.0),
        (True, "True"),
        (True, None),
        (datetime.datetime(2019, 1, 1), True),
        ("2019-01-01", datetime.datetime(2019, 1, 1)),
        # we are allowing object arrays comparison with a boolean value
        pytest.param("True", True, marks=pytest.mark.xfail(reason="see gh-193")),
        pytest.param(b"True", True, marks=pytest.mark.xfail(reason="see gh-193")),
        pytest.param([True], True, marks=pytest.mark.xfail(reason="see gh-193")),
    ],
)
@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">=", "in"])
def test_raise_on_type(value, filter_value, op):
    array_like = pd.Series([value])
    with pytest.raises(TypeError, match="Unexpected type for predicate:"):
        filter_array_like(array_like, op, filter_value, strict_date_types=True)


@pytest.mark.parametrize("op", ["==", "!=", ">=", "<=", ">", "<", "in"])
@pytest.mark.parametrize(
    "data,value",
    [
        (
            # data
            range(10),
            # value
            4,
        ),
        (
            # data
            ["A", "B"] * 5,
            # value
            "A",
        ),
        (
            # data
            pd.Series(["X", "Y"] * 5).astype("category"),
            # value
            "X",
        ),
        (
            # data
            pd.Series([datetime.date(2019, 1, 1), datetime.date(2019, 1, 2)] * 5),
            # value
            datetime.date(2019, 1, 1),
        ),
        (
            # data
            [datetime.datetime(2019, 1, 1), datetime.datetime(2019, 1, 2)] * 5,
            # value
            datetime.datetime(2019, 1, 1),
        ),
        (
            # data
            [datetime.datetime(2019, 1, 1), datetime.datetime(2019, 1, 2)] * 5,
            # value
            datetime.date(2019, 1, 1),
        ),
        (
            # data
            np.arange(10, dtype=np.uint8),
            # value
            4,
        ),
    ],
)
def test_filter_df_from_predicates(op, data, value):
    df = pd.DataFrame({"A": data})
    df["B"] = range(len(df))

    if op == "in":
        value = [value]

    predicates = [[("A", op, value)]]
    actual = filter_df_from_predicates(df, predicates)
    if pd.api.types.is_categorical(df["A"]):
        df["A"] = df["A"].astype(df["A"].cat.as_ordered().dtype)
    if isinstance(value, datetime.date) and (df["A"].dtype == "datetime64[ns]"):
        # silence pandas warning
        value = pd.Timestamp(value)

    if op == "in":
        expected = df[df["A"].isin(value)]
    else:
        expected = eval(f"df[df['A'] {op} value]")
    pdt.assert_frame_equal(actual, expected, check_categorical=False)


@pytest.mark.parametrize("op", ["==", "!="])
@pytest.mark.parametrize("col", list("AB"))
def test_filter_df_from_predicates_bool(op, col):
    df = pd.DataFrame(
        {"A": [True, False] * 5, "B": [True, False, None, True, False] * 2}
    )

    value = True
    predicates = [[(col, op, value)]]
    actual = filter_df_from_predicates(df, predicates)
    if pd.api.types.is_categorical(df[col]):
        df[col] = df[col].astype(df[col].cat.as_ordered().dtype)
    expected = eval(f"df[df[col] {op} value]")
    pdt.assert_frame_equal(actual, expected, check_categorical=False)


@pytest.mark.parametrize(
    "value",
    [
        1,
        np.uint8(1),
        1.1,
        "A",
        datetime.date(2020, 1, 1),
        pd.Timestamp("2020-01-01"),
        np.datetime64("2020-01-01"),
    ],
)
def test_filter_df_from_predicates_empty_in(value):
    df = pd.DataFrame({"A": [value]})
    df["B"] = range(len(df))

    predicates = [[("A", "in", [])]]
    actual = filter_df_from_predicates(df, predicates)
    expected = df.iloc[[]]
    pdt.assert_frame_equal(actual, expected, check_categorical=False)


def test_filter_df_from_predicates_or_predicates():
    df = pd.DataFrame({"A": range(10), "B": ["A", "B"] * 5, "C": range(-10, 0)})

    predicates = [[("A", "<", 3)], [("A", ">", 5)], [("B", "==", "non-existent")]]
    actual = filter_df_from_predicates(df, predicates)
    expected = pd.DataFrame(
        data={
            "A": [0, 1, 2, 6, 7, 8, 9],
            "B": ["A", "B", "A", "A", "B", "A", "B"],
            "C": [-10, -9, -8, -4, -3, -2, -1],
        },
        index=[0, 1, 2, 6, 7, 8, 9],
    )
    pdt.assert_frame_equal(actual, expected)

    predicates = [[("A", "<", 3)], [("A", ">", 5)], [("B", "==", "B")]]
    actual = filter_df_from_predicates(df, predicates)
    # row for (A == 4) is filtered out
    expected = pd.DataFrame(
        data={
            "A": [0, 1, 2, 3, 5, 6, 7, 8, 9],
            "B": ["A", "B", "A", "B", "B", "A", "B", "A", "B"],
            "C": [-10, -9, -8, -7, -5, -4, -3, -2, -1],
        },
        index=[0, 1, 2, 3, 5, 6, 7, 8, 9],
    )
    pdt.assert_frame_equal(actual, expected)
