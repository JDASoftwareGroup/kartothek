from functools import partial

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest
from storefact import get_store_from_url

from kartothek.core.factory import DatasetFactory
from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io_components.utils import (
    align_categories,
    check_single_table_dataset,
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


def test_sort_categorical_pyarrow_conversion():
    """
    Make sure sorting does not introduce indices that end up in the Arrow table.
    """
    df = pd.DataFrame(dict(a=[3, 2, 1]))
    sorted_df = sort_values_categorical(df, "a")
    table = pa.Table.from_pandas(df)
    sorted_table = pa.Table.from_pandas(sorted_df)
    assert table.schema.names == sorted_table.schema.names


def test_check_single_table_factory_nonexistant():
    dataset = DatasetFactory(
        dataset_uuid="dataset_uuid",
        store_factory=partial(get_store_from_url, "hmemory://"),
    )
    check_single_table_dataset(dataset, expected_table="table")


@pytest.fixture(scope="session")
def dataset_single_table(df_all_types, store_session_factory):
    dataset_uuid = gen_uuid()
    return store_dataframes_as_dataset(
        dataset_uuid=dataset_uuid, store=store_session_factory, dfs=df_all_types
    )


@pytest.fixture(scope="session")
def dataset_multi_table(df_all_types, store_session_factory):
    dataset_uuid = gen_uuid()
    return store_dataframes_as_dataset(
        dataset_uuid=dataset_uuid,
        store=store_session_factory,
        dfs={"table1": df_all_types, "table2": df_all_types},
    )


def test_check_single_table_dataset(dataset_single_table):
    check_single_table_dataset(dataset_single_table)
    check_single_table_dataset(dataset_single_table, expected_table="table")
    with pytest.raises(TypeError, match="Unexpected table in dataset"):
        check_single_table_dataset(
            dataset_single_table, expected_table="table_not_found"
        )


def test_check_single_table_factory(dataset_single_table, store_session_factory):
    dataset_factory = DatasetFactory(
        dataset_uuid=dataset_single_table.uuid, store_factory=store_session_factory
    )
    check_single_table_dataset(dataset_factory)
    check_single_table_dataset(dataset_factory, expected_table="table")
    with pytest.raises(TypeError, match="Unexpected table in dataset"):
        check_single_table_dataset(dataset_factory, expected_table="table_not_found")


def test_check_single_table_multi_raises(dataset_multi_table):
    with pytest.raises(
        TypeError, match="Expected single table dataset but found dataset with tables:"
    ):
        check_single_table_dataset(
            dataset_multi_table, expected_table="table_not_found"
        )
    with pytest.raises(
        TypeError, match="Expected single table dataset but found dataset with tables:"
    ):
        check_single_table_dataset(dataset_multi_table)
