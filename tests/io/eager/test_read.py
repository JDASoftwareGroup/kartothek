import datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.io.eager import (
    read_dataset_as_dataframes,
    read_dataset_as_metapartitions,
    read_table,
    store_dataframes_as_dataset,
)
from kartothek.io.testing.read import *  # noqa
from kartothek.io_components.metapartition import SINGLE_TABLE


@pytest.fixture(
    params=["dataframe", "metapartition", "table"],
    ids=["dataframe", "metapartition", "table"],
)
def output_type(request):
    # TODO: get rid of this parametrization and split properly into two functions
    return request.param


def _read_table(*args, **kwargs):
    if "tables" in kwargs:
        param_tables = kwargs.pop("tables")
        kwargs["table"] = param_tables
    res = read_table(*args, **kwargs)

    if len(res):
        # Array split conserves dtypes
        return np.array_split(res, len(res))
    else:
        return [res]


def _read_dataset(output_type, *args, **kwargs):
    if output_type == "table":
        return _read_table
    elif output_type == "dataframe":
        return read_dataset_as_dataframes
    elif output_type == "metapartition":
        return read_dataset_as_metapartitions
    else:
        raise NotImplementedError()


@pytest.fixture()
def bound_load_dataframes(output_type):
    return _read_dataset(output_type)


@pytest.fixture()
def backend_identifier():
    return "eager"


def test_read_table_eager(dataset, store_session, use_categoricals):
    if use_categoricals:
        categories = {SINGLE_TABLE: ["P"]}
    else:
        categories = None

    df = read_table(
        store=store_session,
        dataset_uuid="dataset_uuid",
        table=SINGLE_TABLE,
        categoricals=categories,
    )
    expected_df = pd.DataFrame(
        {
            "P": [1, 2],
            "L": [1, 2],
            "TARGET": [1, 2],
            "DATE": pd.to_datetime(
                [datetime.date(2010, 1, 1), datetime.date(2009, 12, 31)]
            ),
        }
    )
    if categories:
        expected_df = expected_df.astype({"P": "category"})

    # No stability of partitions
    df = df.sort_values(by="P").reset_index(drop=True)

    pdt.assert_frame_equal(df, expected_df, check_dtype=True, check_like=True)

    df_2 = read_table(store=store_session, dataset_uuid="dataset_uuid", table="helper")
    expected_df_2 = pd.DataFrame({"P": [1, 2], "info": ["a", "b"]})

    assert isinstance(df_2, pd.DataFrame)

    # No stability of partitions
    df_2 = df_2.sort_values(by="P").reset_index(drop=True)

    pdt.assert_frame_equal(df_2, expected_df_2, check_dtype=True, check_like=True)

    df_3 = read_table(
        store=store_session,
        dataset_uuid="dataset_uuid",
        table="helper",
        predicates=[[("P", "==", 2)]],
    )
    expected_df_3 = pd.DataFrame({"P": [2], "info": ["b"]})

    assert isinstance(df_3, pd.DataFrame)
    pdt.assert_frame_equal(df_3, expected_df_3, check_dtype=True, check_like=True)

    df_4 = read_table(
        store=store_session,
        dataset_uuid="dataset_uuid",
        table="helper",
        predicates=[[("info", "==", "a")]],
    )
    expected_df_4 = pd.DataFrame({"P": [1], "info": ["a"]})

    assert isinstance(df_4, pd.DataFrame)

    pdt.assert_frame_equal(df_4, expected_df_4, check_dtype=True, check_like=True)


def test_read_table_with_columns(dataset, store_session):
    df = read_table(
        store=store_session,
        dataset_uuid="dataset_uuid",
        table=SINGLE_TABLE,
        columns=["P", "L"],
    )

    expected_df = pd.DataFrame({"P": [1, 2], "L": [1, 2]})

    # No stability of partitions
    df = df.sort_values(by="P").reset_index(drop=True)
    expected_df = expected_df.sort_values(by="P").reset_index(drop=True)

    pdt.assert_frame_equal(df, expected_df, check_dtype=False, check_like=True)


def test_read_table_simple_list_for_cols_cats(dataset, store_session):
    df = read_table(
        store=store_session,
        dataset_uuid="dataset_uuid",
        table=SINGLE_TABLE,
        columns=["P", "L"],
        categoricals=["P", "L"],
    )

    expected_df = pd.DataFrame({"P": [1, 2], "L": [1, 2]})

    # No stability of partitions
    df = df.sort_values(by="P").reset_index(drop=True)
    expected_df = expected_df.sort_values(by="P").reset_index(drop=True)

    expected_df = expected_df.astype("category")

    pdt.assert_frame_equal(df, expected_df, check_dtype=False, check_like=True)


def test_write_and_read_default_table_name(store_session):

    df_write = pd.DataFrame({"P": [3.14, 2.71]})
    store_dataframes_as_dataset(store_session, "test_default_table_name", [df_write])

    # assert default table name "table" is used
    df_read_as_dfs = read_dataset_as_dataframes(
        "test_default_table_name", store_session
    )
    pd.testing.assert_frame_equal(df_write, df_read_as_dfs[0]["table"])


@pytest.mark.parametrize("partition_on", [None, ["A", "B"]])
def test_read_or_predicates(store_factory, partition_on):
    # https://github.com/JDASoftwareGroup/kartothek/issues/295
    dataset_uuid = "test"
    df = pd.DataFrame({"A": range(10), "B": ["A", "B"] * 5, "C": range(-10, 0)})

    store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        dfs=[df],
        partition_on=partition_on,
    )

    df1 = read_table(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        predicates=[[("A", "<", 3)], [("A", ">", 5)], [("B", "==", "non-existent")]],
    )

    df2 = read_table(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        predicates=[[("A", "<", 3)], [("A", ">", 5)]],
    )
    expected = pd.DataFrame(
        data={
            "A": [0, 1, 2, 6, 7, 8, 9],
            "B": ["A", "B", "A", "A", "B", "A", "B"],
            "C": [-10, -9, -8, -4, -3, -2, -1],
        },
    )

    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(expected, df2)
