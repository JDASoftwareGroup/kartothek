import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.utils.pandas import (
    aggregate_to_lists,
    concat_dataframes,
    drop_sorted_duplicates_keep_last,
    is_dataframe_sorted,
    mask_sorted_duplicates_keep_last,
    merge_dataframes_robust,
    sort_dataframe,
)


class TestConcatDataframes:
    @pytest.fixture(params=[True, False])
    def dummy_default(self, request):
        if request.param:
            return pd.DataFrame(data={"a": [-2, -3], "b": 1.0}, columns=["a", "b"])
        else:
            return None

    @pytest.fixture(params=[True, False])
    def maybe_iter(self, request):
        if request.param:
            return iter
        else:
            return list

    def test_many(self, dummy_default, maybe_iter):
        dfs = [
            pd.DataFrame(
                data={"a": [0, 1], "b": 1.0}, columns=["a", "b"], index=[10, 11]
            ),
            pd.DataFrame(
                data={"a": [2, 3], "b": 2.0}, columns=["a", "b"], index=[10, 11]
            ),
            pd.DataFrame(data={"a": [4, 5], "b": 3.0}, columns=["a", "b"]),
        ]
        expected = pd.DataFrame(
            {"a": [0, 1, 2, 3, 4, 5], "b": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]},
            columns=["a", "b"],
        )

        actual = concat_dataframes(maybe_iter(dfs), dummy_default)
        pdt.assert_frame_equal(actual, expected)

    def test_single(self, dummy_default, maybe_iter):
        df = pd.DataFrame(
            data={"a": [0, 1], "b": 1.0}, columns=["a", "b"], index=[10, 11]
        )

        actual = concat_dataframes(maybe_iter([df.copy()]), dummy_default)
        pdt.assert_frame_equal(actual, df)

    def test_default(self, maybe_iter):
        df = pd.DataFrame(
            data={"a": [0, 1], "b": 1.0}, columns=["a", "b"], index=[10, 11]
        )

        actual = concat_dataframes(maybe_iter([]), df)
        pdt.assert_frame_equal(actual, df)

    def test_fail_no_default(self, maybe_iter):
        with pytest.raises(ValueError) as exc:
            concat_dataframes(maybe_iter([]), None)
        assert str(exc.value) == "Cannot concatenate 0 dataframes."

    @pytest.mark.parametrize(
        "dfs",
        [
            [pd.DataFrame({"a": [0, 1]})],
            [pd.DataFrame({"a": [0, 1]}), pd.DataFrame({"a": [2, 3]})],
        ],
    )
    def test_whipe_list(self, dfs):
        concat_dataframes(dfs)
        assert dfs == []

    @pytest.mark.parametrize(
        "dfs,expected",
        [
            (
                # dfs
                [pd.DataFrame(index=range(3))],
                # expected
                pd.DataFrame(index=range(3)),
            ),
            (
                # dfs
                [pd.DataFrame(index=range(3)), pd.DataFrame(index=range(2))],
                # expected
                pd.DataFrame(index=range(5)),
            ),
        ],
    )
    def test_no_columns(self, dfs, expected):
        actual = concat_dataframes(dfs)
        pdt.assert_frame_equal(actual, expected)

    def test_fail_different_colsets(self, maybe_iter):
        dfs = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1], "b": [2]})]
        with pytest.raises(
            ValueError, match="Not all DataFrames have the same set of columns!"
        ):
            concat_dataframes(maybe_iter(dfs))


@pytest.mark.parametrize(
    "df,columns",
    [
        (
            # df
            pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3]}),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame({"a": [3, 2, 1, 3, 2, 1], "b": [2, 2, 2, 1, 1, 1]}),
            # columns
            ["a", "b"],
        ),
        (
            # df
            pd.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3]}, index=[1000, 2000, 3000]),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame({"a": [3.0, 2.0, 1.0], "b": [1, 2, 3]}),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame({"a": ["3", "2", "1"], "b": [1, 2, 3]}),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame({"a": [True, False], "b": [1, 2]}),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame(
                {
                    "a": [
                        pd.Timestamp("2018-01-03"),
                        pd.Timestamp("2018-01-02"),
                        pd.Timestamp("2018-01-01"),
                    ],
                    "b": [1, 2, 3],
                }
            ),
            # columns
            ["a"],
        ),
        (
            # df
            pd.DataFrame(
                {"a": pd.Series(["3", "2", "1"]).astype("category"), "b": [1, 2, 3]}
            ),
            # columns
            ["a"],
        ),
    ],
)
def test_sort_dataframe(df, columns):
    expected = df.sort_values(columns).reset_index(drop=True)
    actual = sort_dataframe(df, columns)
    pdt.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "df,columns,expected_mask",
    [
        (
            # df
            pd.DataFrame({"a": [1, 2, 3]}),
            # columns
            ["a"],
            # expected_mask
            np.array([False, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 1, 3]}),
            # columns
            ["a"],
            # expected_mask
            np.array([True, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 1, 3], "b": [1, 2, 3]}),
            # columns
            ["a"],
            # expected_mask
            np.array([True, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 1, 3], "b": [1, 2, 3]}),
            # columns
            ["a", "b"],
            # expected_mask
            np.array([False, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 1, 3], "b": [1, 1, 3]}),
            # columns
            ["a", "b"],
            # expected_mask
            np.array([True, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1]}),
            # columns
            ["a"],
            # expected_mask
            np.array([False]),
        ),
        (
            # df
            pd.DataFrame({"a": []}),
            # columns
            ["a"],
            # expected_mask
            np.array([], dtype=bool),
        ),
        (
            # df
            pd.DataFrame(
                {
                    "a": [1, 1, 3],
                    "b": [1.0, 1.0, 3.0],
                    "c": ["a", "a", "b"],
                    "d": [True, True, False],
                    "e": [
                        pd.Timestamp("2018"),
                        pd.Timestamp("2018"),
                        pd.Timestamp("2019"),
                    ],
                    "f": pd.Series(["a", "a", "b"]).astype("category"),
                }
            ),
            # columns
            ["a", "b", "c", "d", "e", "f"],
            # expected_mask
            np.array([True, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 2, 3, 4, 4, 5, 6, 6]}),
            # columns
            ["a"],
            # expected_mask
            np.array([False, False, False, True, False, False, True, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [2, 2, 3]}),
            # columns
            [],
            # expected_mask
            np.array([False, False, False]),
        ),
        (
            # df
            pd.DataFrame({"a": [1, 1, 3]}, index=[1000, 2000, 1]),
            # columns
            ["a"],
            # expected_mask
            np.array([True, False, False]),
        ),
    ],
)
def test_sorted_duplicates_keep_last(df, columns, expected_mask):
    actual_mask = mask_sorted_duplicates_keep_last(df, columns)
    assert actual_mask.dtype == bool
    npt.assert_array_equal(actual_mask, expected_mask)

    actual_df = drop_sorted_duplicates_keep_last(df, columns)
    expected_df = df[~expected_mask]
    pdt.assert_frame_equal(actual_df, expected_df)

    if columns:
        # pandas crashes for empty column lists
        pd_mask = df.duplicated(subset=columns, keep="last").values
        pd_df = df.drop_duplicates(subset=columns, keep="last")
        npt.assert_array_equal(pd_mask, expected_mask)
        pdt.assert_frame_equal(pd_df, expected_df)


@pytest.mark.parametrize(
    "df_input,by",
    [
        (
            # df_input
            pd.DataFrame({"x": [0], "y": [0], "v": ["a"]}),
            # by
            ["x", "y"],
        ),
        (
            # df_input
            pd.DataFrame({"x": [0, 0], "y": [0, 0], "v": ["a", "b"]}),
            # by
            ["x", "y"],
        ),
        (
            # df_input
            pd.DataFrame({"x": [0, 0], "y": [0, 0], "v": ["a", "a"]}),
            # by
            ["x", "y"],
        ),
        (
            # df_input
            pd.DataFrame(
                {
                    "x": [1, 0, 0, 1, 1],
                    "y": [1, 0, 0, 0, 1],
                    "v": ["a", "b", "c", "d", "e"],
                }
            ),
            # by
            ["x", "y"],
        ),
        (
            # df_input
            pd.DataFrame({"x": [], "y": [], "v": []}),
            # by
            ["x", "y"],
        ),
        (
            # df_input
            pd.DataFrame({"x": [0, 0], "y": [0, 0], "v": ["a", "a"]}),
            # by
            [],
        ),
    ],
)
def test_aggregate_to_lists(df_input, by):
    data_col = "v"

    # pandas is broken for empty DFs
    if df_input.empty:
        df_expected = df_input
    else:
        if by:
            df_expected = df_input.groupby(by=by, as_index=False)[data_col].agg(
                lambda series: list(series.values)
            )
        else:
            df_expected = pd.DataFrame(
                {data_col: pd.Series([list(df_input[data_col].values)])}
            )

    df_actual = aggregate_to_lists(df_input, by, data_col)

    pdt.assert_frame_equal(df_actual, df_expected)


def test_is_dataframe_sorted_no_cols():
    df = pd.DataFrame({})
    with pytest.raises(ValueError, match="`columns` must contain at least 1 column"):
        is_dataframe_sorted(df, [])


@pytest.mark.parametrize(
    "df,columns",
    [
        (
            # df
            pd.DataFrame({"x": []}),
            # columns
            ["x"],
        ),
        (
            # df
            pd.DataFrame({"x": [], "y": [], "z": []}),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 1, 10]}),
            # columns
            ["x"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 1, 10], "y": [20, 21, 210]}),
            # columns
            ["x", "y"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 1, 1], "y": [10, 0, 1]}),
            # columns
            ["x", "y"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 1], "y": [1, 0]}),
            # columns
            ["x"],
        ),
        (
            # df
            pd.DataFrame(
                {
                    "x": [0, 0, 0, 0, 1, 1, 1, 1],
                    "y": [0, 0, 1, 1, 0, 0, 1, 1],
                    "z": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            ),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 0], "y": [0, 0], "z": [0, 0]}),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": pd.Series(["1", "2"]).astype("category")}),
            # columns
            ["x"],
        ),
    ],
)
def test_assert_df_sorted_ok(df, columns):
    assert is_dataframe_sorted(df, columns)


@pytest.mark.parametrize(
    "df,columns",
    [
        (
            # df
            pd.DataFrame({"x": [1, 0]}),
            # columns
            ["x"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 1, 1], "y": [0, 1, 0]}),
            # columns
            ["x", "y"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 0], "y": [0, 0], "z": [1, 0]}),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": [0, 0], "y": [1, 0], "z": [0, 0]}),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": [1, 0], "y": [0, 0], "z": [0, 0]}),
            # columns
            ["x", "y", "z"],
        ),
        (
            # df
            pd.DataFrame({"x": pd.Series(["2", "1"]).astype("category")}),
            # columns
            ["x"],
        ),
    ],
)
def test_assert_df_sorted_no(df, columns):
    assert not is_dataframe_sorted(df, columns)


@pytest.mark.parametrize(
    "df1,df2,how,expected",
    [
        (
            # df1
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"x": [0, 1], "v2": [21, 22]}),
            # how
            "inner",
            # expected
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12], "v2": [21, 22]}),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"x": [0, 1], "v2": [21, 22]}),
            # how
            "left",
            # expected
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12], "v2": [21, 22]}),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"x": [0], "v2": [21]}),
            # how
            "inner",
            # expected
            pd.DataFrame({"i": [0], "x": [0], "v1": [11], "v2": [21]}),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "x": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"x": [0], "v2": [21]}),
            # how
            "left",
            # expected
            pd.DataFrame(
                {"i": [0, 1], "x": [0, 1], "v1": [11, 12], "v2": [21, np.nan]}
            ),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"v2": [21, 22]}),
            # how
            "inner",
            # expected
            pd.DataFrame(
                {"i": [0, 0, 1, 1], "v1": [11, 11, 12, 12], "v2": [21, 22, 21, 22]}
            ),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"v2": [21, 22]}),
            # how
            "left",
            # expected
            pd.DataFrame(
                {"i": [0, 0, 1, 1], "v1": [11, 11, 12, 12], "v2": [21, 22, 21, 22]}
            ),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"v2": pd.Series([], dtype=int)}),
            # how
            "inner",
            # expected
            pd.DataFrame(
                {
                    "i": pd.Series([], dtype=int),
                    "v1": pd.Series([], dtype=int),
                    "v2": pd.Series([], dtype=int),
                }
            ),
        ),
        (
            # df1
            pd.DataFrame({"i": [0, 1], "v1": [11, 12]}),
            # df2
            pd.DataFrame({"v2": pd.Series([], dtype=int)}),
            # how
            "left",
            # expected
            pd.DataFrame({"i": [0, 1], "v1": [11, 12], "v2": [np.nan, np.nan]}),
        ),
    ],
)
def test_merge_dataframes_robust(df1, df2, how, expected):
    df1_backup = df1.copy()
    df2_backup = df2.copy()

    actual = merge_dataframes_robust(df1, df2, how)
    actual = actual.sort_values(sorted(actual.columns)).reset_index(drop=True)

    pdt.assert_frame_equal(df1, df1_backup)
    pdt.assert_frame_equal(df2, df2_backup)

    pdt.assert_frame_equal(actual, expected)
