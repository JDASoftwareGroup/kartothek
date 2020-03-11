from datetime import date

import dask
import pandas as pd
import pandas.testing as pdt

from kartothek.io.dask.delayed import merge_many_datasets_as_delayed


def _merge_many(dfs, *args, **kwargs):
    return pd.merge(dfs[0], dfs[1], *args, **kwargs)


MERGE_TASKS = [
    {
        "tables": ["table", "PRED"],
        "merge_func": _merge_many,
        "merge_kwargs": {"how": "left", "sort": False, "copy": False},
        "output_label": "merged_core_data",
    }
]

MERGE_EXP_CL1 = pd.DataFrame(
    {
        "P": [1],
        "L": [1],
        "TARGET": [1],
        "HORIZON": [1],
        "PRED": [10],
        "DATE": pd.to_datetime([date(2010, 1, 1)]),
    }
)

MERGE_EXP_CL2 = pd.DataFrame(
    {
        "P": [2],
        "L": [2],
        "TARGET": [2],
        "HORIZON": [1],
        "PRED": [10],
        "DATE": pd.to_datetime([date(2009, 12, 31)]),
    }
)


def test_merge_many_datasets_prefix_first(
    dataset, evaluation_dataset, store_factory, store_session_factory, frozen_time
):
    df_list = merge_many_datasets_as_delayed(
        dataset_uuids=[dataset.uuid, evaluation_dataset.uuid],
        store=store_session_factory,
        merge_tasks=MERGE_TASKS,
        match_how="prefix_first",
    )
    df_list = dask.compute(df_list)[0]
    df_list = [mp.data for mp in df_list]

    # Two partitions
    assert len(df_list) == 2
    assert len(df_list[1]) == 2
    assert len(df_list[0]) == 2
    pdt.assert_frame_equal(
        df_list[0]["merged_core_data"],
        MERGE_EXP_CL1,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
    pdt.assert_frame_equal(
        df_list[1]["merged_core_data"],
        MERGE_EXP_CL2,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )


MERGE_TASKS_FIRST = [
    {
        "tables": ["table_0", "table_1"],
        "merge_func": _merge_many,
        "merge_kwargs": {"how": "outer", "sort": False, "copy": False},
        "output_label": "merged_core_data",
    }
]

MERGE_EXP_CL1_FIRST = pd.DataFrame(
    {"P": [1], "L": [1], "TARGET": [1], "DATE": pd.to_datetime([date(2010, 1, 1)])}
)

MERGE_EXP_CL2_FIRST = pd.DataFrame(
    {
        "P": [1, 2],
        "L": [1, 2],
        "TARGET": [1, 2],
        "DATE": pd.to_datetime([date(2010, 1, 1), date(2009, 12, 31)]),
    }
)
MERGE_EXP_CL3_FIRST = pd.DataFrame(
    {
        "P": [2, 1],
        "L": [2, 1],
        "TARGET": [2, 1],
        "DATE": pd.to_datetime([date(2009, 12, 31), date(2010, 1, 1)]),
    }
)
MERGE_EXP_CL4_FIRST = pd.DataFrame(
    {"P": [2], "L": [2], "TARGET": [2], "DATE": pd.to_datetime([date(2009, 12, 31)])}
)


def test_merge_many_dataset_first(
    dataset_partition_keys, store_session_factory, frozen_time
):
    df_list = merge_many_datasets_as_delayed(
        dataset_uuids=[dataset_partition_keys.uuid, dataset_partition_keys.uuid],
        store=store_session_factory,
        merge_tasks=MERGE_TASKS_FIRST,
        match_how="first",
    )
    df_list = dask.compute(df_list)[0]
    df_list = [mp.data for mp in df_list]
    assert len(df_list) == 4
    pdt.assert_frame_equal(
        df_list[0]["merged_core_data"],
        MERGE_EXP_CL1_FIRST,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
    pdt.assert_frame_equal(
        df_list[1]["merged_core_data"],
        MERGE_EXP_CL2_FIRST,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
    pdt.assert_frame_equal(
        df_list[2]["merged_core_data"],
        MERGE_EXP_CL3_FIRST,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
    pdt.assert_frame_equal(
        df_list[3]["merged_core_data"],
        MERGE_EXP_CL4_FIRST,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )


def test_merge_many_dataset_dispatch_by(
    dataset_partition_keys, store_session_factory, frozen_time
):
    df_list = merge_many_datasets_as_delayed(
        dataset_uuids=[dataset_partition_keys.uuid, dataset_partition_keys.uuid],
        store=store_session_factory,
        merge_tasks=MERGE_TASKS_FIRST,
        match_how="dispatch_by",
        dispatch_by=["P"],
    )
    df_list = dask.compute(df_list)[0]
    df_list = [mp.data for mp in df_list]
