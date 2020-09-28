from functools import partial
from typing import Callable

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest
from storefact import get_store_from_url

from kartothek.io_components.utils import (
    align_categories,
    combine_metadata,
    extract_duplicates,
    normalize_args,
    sort_values_categorical,
)


def testcombine_metadata():
    d1 = {"a": 1, "b": 1}
    d2 = {"a": 1, "b": 2}
    d3 = {"a": 2, "b": 2}
    d4 = {"a": 1, "c": 3}
    assert combine_metadata([d1]) == d1
    assert combine_metadata([d1, d1]) == d1
    assert combine_metadata([d1, d2]) == {"a": 1}
    assert combine_metadata([d1, d2, d3]) == {}
    assert combine_metadata([d1, d4]) == {"a": 1, "b": 1, "c": 3}


def testcombine_metadata_incompatible():
    d1 = {"a": 1, "b": [1]}
    d2 = {"a": 1, "b": 1}
    assert combine_metadata([d1, d2]) == {"a": 1}


def testcombine_metadata_nested():
    d1 = {"a": {"l2a": 1}, "b": {"l2b": 1}}
    d2 = {"a": {"l2a": 1}, "b": {"l2b": 2}}
    d3 = {"a": {"l2a": 2}, "b": {"l2b": 2}}
    d4 = {"a": {"l2a": 1}, "c": {"l2c": 3}}
    assert combine_metadata([d1]) == d1
    assert combine_metadata([d1, d1]) == d1
    assert combine_metadata([d1, d2]) == {"a": {"l2a": 1}}
    assert combine_metadata([d1, d2, d3]) == {}
    assert combine_metadata([d1, d4]) == {
        "a": {"l2a": 1},
        "b": {"l2b": 1},
        "c": {"l2c": 3},
    }


def testcombine_metadata_nested_lists():
    d1 = {"a": [[1], [2]], "b": [[1], [2]]}
    assert combine_metadata([d1]) == d1

    assert combine_metadata([d1, d1]) == {"a": [[1], [2]], "b": [[1], [2]]}


def testcombine_metadata_lists():
    d1 = {"a": [1, 2], "b": [1, 2]}
    assert combine_metadata([d1], append_to_list=False) == d1

    combined_metadata = combine_metadata([d1, d1], append_to_list=False)
    assert set(combined_metadata["a"]) == set([1, 2])
    assert set(combined_metadata["b"]) == set([1, 2])


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("a", "a", "a"), ("a", ["a"], ["a"])),
        (("a", "abc", "a"), ("a", ["abc"], ["a"])),
        (("a", "a", "abc"), ("a", ["a"], ["abc"])),
        (("a", "abc", "abc"), ("a", ["abc"], ["abc"])),
        (("a", "abc", None), ("a", ["abc"], [])),
        (("a", None, "abc"), ("a", [], ["abc"])),
        (("a", None, None), ("a", [], [])),
        (("a", (1, 2), ("a", "b")), ("a", [1, 2], ["a", "b"])),
    ],
)
def test_normalize_args(test_input, expected):
    @normalize_args
    def func(arg1, partition_on, delete_scope=None):
        return arg1, partition_on, delete_scope

    test_arg1, test_partition_on, test_delete_scope = test_input
    assert expected == func(
        test_arg1, test_partition_on, delete_scope=test_delete_scope
    )
    assert (expected[0], expected[1], []) == func(test_arg1, test_partition_on)


@pytest.mark.parametrize("_type", ["callable", "url", "simplekv"])
def test_normalize_store(tmpdir, _type):

    store_url = f"hfs://{tmpdir}"
    store = get_store_from_url(store_url)
    store.put("test", b"")

    @normalize_args
    def func(store):
        assert isinstance(store, Callable)
        return store().keys()

    if _type == "callable":
        store_test = partial(get_store_from_url, store_url)
    elif _type == "url":
        store_test = store_url
    elif _type == "simplekv":
        store_test = store
    else:
        raise AssertionError(f"unknown parametrization {_type}")
    assert func(store_test)


@pytest.mark.parametrize(
    "test_input", [("a", {"a"}, "c"), ("a", frozenset("a"), None), ("abc", {"c": 6}, 4)]
)
def test_normalize_args__incompatible_types(test_input):
    @normalize_args
    def func(arg1, partition_on, delete_scope=None):
        return arg1, partition_on, delete_scope

    test_arg1, test_partition_on, test_delete_scope = test_input
    with pytest.raises(ValueError):
        func(test_arg1, test_partition_on, delete_scope=test_delete_scope)


@pytest.mark.parametrize(
    "lst, expected",
    [
        ([], []),
        (["a"], []),
        (["a", "b", "c"], []),
        (["a", "b", "c", "a"], ["a"]),
        (["a", "b", "c", "a", "a"], ["a"]),
        (["a", "a"], ["a"]),
    ],
)
def test_extract_duplicates(lst, expected):
    assert set(extract_duplicates(lst)) == set(expected)


def test_align_categories():
    df1 = pd.DataFrame(
        {
            "col_A": pd.Categorical(["A1", "A3", "A3"]),
            "col_B": pd.Categorical(["B1", "B3", "B3"]),
        }
    )
    df2 = pd.DataFrame(
        {
            "col_A": pd.Categorical(["A2", "A3", "A4"]),
            "col_B": pd.Categorical(["B2", "B3", "B4"]),
        }
    )
    df3 = pd.DataFrame(
        {
            "col_A": pd.Categorical(["A4", "A5", "A1"]),
            "col_B": pd.Categorical(["B4", "B5", "B1"]),
        }
    )
    in_dfs = [df1, df2, df3]

    out_dfs = align_categories(in_dfs, categoricals=["col_A", "col_B"])

    for prefix in ["A", "B"]:
        col_name = "col_{}".format(prefix)
        expected_categories = [
            "{}1".format(prefix),
            "{}3".format(prefix),
            "{}2".format(prefix),
            "{}4".format(prefix),
            "{}5".format(prefix),
        ]
        expected_1 = pd.Series(
            pd.Categorical(
                ["{}1".format(prefix), "{}3".format(prefix), "{}3".format(prefix)],
                categories=expected_categories,
            ),
            name=col_name,
        )
        pdt.assert_series_equal(out_dfs[0][col_name], expected_1)

        expected_2 = pd.Series(
            pd.Categorical(
                ["{}2".format(prefix), "{}3".format(prefix), "{}4".format(prefix)],
                categories=expected_categories,
            ),
            name=col_name,
        )
        pdt.assert_series_equal(out_dfs[1][col_name], expected_2)

        expected_3 = pd.Series(
            pd.Categorical(
                ["{}4".format(prefix), "{}5".format(prefix), "{}1".format(prefix)],
                categories=expected_categories,
            ),
            name=col_name,
        )
        pdt.assert_series_equal(out_dfs[2][col_name], expected_3)


def test_sort_cateogrical():
    values = ["f", "a", "b", "z", "e"]
    categories = ["e", "z", "b", "a", "f"]
    cat_ser = pd.Series(pd.Categorical(values, categories=categories))
    df = pd.DataFrame({"cat": cat_ser, "int": range(len(values))})

    sorted_df = sort_values_categorical(df, "cat")

    expected_values = sorted(values)

    assert all(sorted_df["cat"].values == expected_values)
    assert sorted_df["cat"].is_monotonic
    assert sorted_df["cat"].cat.ordered
    assert all(sorted_df["cat"].cat.categories == sorted(categories))


def test_sort_cateogrical_multiple_cols():
    df = pd.DataFrame.from_records(
        [
            {"ColA": "B", "ColB": "Z", "Payload": 1},
            {"ColA": "B", "ColB": "A", "Payload": 2},
            {"ColA": "A", "ColB": "A", "Payload": 3},
            {"ColA": "C", "ColB": "Z", "Payload": 4},
        ]
    )
    df_expected = df.copy().sort_values(by=["ColA", "ColB"]).reset_index(drop=True)
    df = df.astype({col: "category" for col in ["ColA", "ColB"]})
    # Correct order
    # {"ColA": "A", "ColB": "A", "Payload": 3},
    # {"ColA": "B", "ColB": "A", "Payload": 2},
    # {"ColA": "B", "ColB": "Z", "Payload": 1},
    # {"ColA": "C", "ColB": "Z", "Payload": 4},
    df_expected = df_expected.astype(
        {
            "ColA": pd.CategoricalDtype(categories=["A", "B", "C"], ordered=True),
            "ColB": pd.CategoricalDtype(categories=["A", "Z"], ordered=True),
        }
    )

    sorted_df = sort_values_categorical(df, ["ColA", "ColB"])

    pdt.assert_frame_equal(sorted_df, df_expected)


def test_sort_categorical_pyarrow_conversion():
    """
    Make sure sorting does not introduce indices that end up in the Arrow table.
    """
    df = pd.DataFrame(dict(a=[3, 2, 1]))
    sorted_df = sort_values_categorical(df, "a")
    table = pa.Table.from_pandas(df)
    sorted_table = pa.Table.from_pandas(sorted_df)
    assert table.schema.names == sorted_table.schema.names
