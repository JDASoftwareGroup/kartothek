from datetime import timedelta
from functools import partial
from itertools import permutations

import dask.bag as db
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kartothek.core.cube.conditions import (
    C,
    Conjunction,
    EqualityCondition,
    GreaterEqualCondition,
    GreaterThanCondition,
    InequalityCondition,
    InIntervalCondition,
    IsInCondition,
    LessEqualCondition,
    LessThanCondition,
)
from kartothek.core.cube.cube import Cube
from kartothek.io.dask.bag_cube import build_cube_from_bag
from kartothek.io.eager import build_dataset_indices
from kartothek.io.eager_cube import append_to_cube, build_cube, remove_partitions

__all__ = (
    "apply_condition_unsafe",
    "data_no_part",
    "fullrange_cube",
    "fullrange_data",
    "fullrange_df",
    "massive_partitions_cube",
    "massive_partitions_data",
    "massive_partitions_df",
    "multipartition_cube",
    "multipartition_df",
    "no_part_cube",
    "no_part_df",
    "other_part_cube",
    "sparse_outer_cube",
    "sparse_outer_data",
    "sparse_outer_df",
    "sparse_outer_opt_cube",
    "sparse_outer_opt_df",
    "test_complete",
    "test_condition",
    "test_condition_on_null",
    "test_cube",
    "test_delayed_index_build_correction_restriction",
    "test_delayed_index_build_partition_by",
    "test_df",
    "test_fail_blocksize_negative",
    "test_fail_blocksize_wrong_type",
    "test_fail_blocksize_zero",
    "test_fail_empty_dimension_columns",
    "test_fail_missing_condition_columns",
    "test_fail_missing_dimension_columns",
    "test_fail_missing_partition_by",
    "test_fail_missing_payload_columns",
    "test_fail_no_store_factory",
    "test_fail_projection",
    "test_fail_unindexed_partition_by",
    "test_fail_unstable_dimension_columns",
    "test_fail_unstable_partition_by",
    "test_filter_select",
    "test_hypothesis",
    "test_overlay_tricky",
    "test_partition_by",
    "test_projection",
    "test_select",
    "test_simple_roundtrip",
    "test_sort",
    "test_stresstest_index_select_row",
    "test_wrong_condition_type",
    "testset",
    "updated_cube",
    "updated_df",
)


@pytest.fixture(scope="module")
def fullrange_data():
    return {
        "seed": pd.DataFrame(
            {
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v1": np.arange(16),
                "i1": np.arange(16),
            }
        ),
        "enrich_dense": pd.DataFrame(
            {
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v2": np.arange(16),
                "i2": np.arange(16),
            }
        ),
        "enrich_sparse": pd.DataFrame(
            {
                "y": [0, 1, 2, 3, 0, 1, 2, 3],
                "z": 0,
                "p": [0, 0, 1, 1, 0, 0, 1, 1],
                "q": [0, 0, 0, 0, 1, 1, 1, 1],
                "v3": np.arange(8),
                "i3": np.arange(8),
            }
        ),
    }


@pytest.fixture(scope="module")
def fullrange_cube(module_store, fullrange_data):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="fullrange_cube",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(data=fullrange_data, store=module_store, cube=cube)
    return cube


@pytest.fixture(scope="module")
def multipartition_cube(module_store, fullrange_data, fullrange_cube):
    def _gen(part):
        result = {}
        for dataset_id, df in fullrange_data.items():
            df = df.copy()
            df["z"] = part
            result[dataset_id] = df
        return result

    cube = fullrange_cube.copy(uuid_prefix="multipartition_cube")
    build_cube_from_bag(
        data=db.from_sequence([0, 1], partition_size=1).map(_gen),
        store=module_store,
        cube=cube,
        ktk_cube_dataset_ids=["seed", "enrich_dense", "enrich_sparse"],
    ).compute()
    return cube


@pytest.fixture(scope="module")
def sparse_outer_data():
    return {
        "seed": pd.DataFrame(
            {
                "x": [0, 1, 0],
                "y": [0, 0, 1],
                "z": 0,
                "p": [0, 1, 2],
                "q": 0,
                "v1": [0, 3, 7],
                "i1": [0, 3, 7],
            }
        ),
        "enrich_dense": pd.DataFrame(
            {
                "x": [0, 0],
                "y": [0, 1],
                "z": 0,
                "p": [0, 2],
                "q": 0,
                "v2": [0, 7],
                "i2": [0, 7],
            }
        ),
        "enrich_sparse": pd.DataFrame(
            {"y": [0, 0], "z": 0, "p": [0, 1], "q": 0, "v3": [0, 3], "i3": [0, 3]}
        ),
    }


@pytest.fixture(scope="module")
def sparse_outer_cube(module_store, sparse_outer_data):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="sparse_outer_cube",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(data=sparse_outer_data, store=module_store, cube=cube)
    return cube


@pytest.fixture(scope="module")
def sparse_outer_opt_cube(
    module_store,
    sparse_outer_data,
    sparse_outer_cube,
    sparse_outer_df,
    sparse_outer_opt_df,
):
    data = {}
    for dataset_id in sparse_outer_data.keys():
        df = sparse_outer_data[dataset_id].copy()

        for col in sparse_outer_opt_df.columns:
            if col in df.columns:
                dtype = sparse_outer_opt_df[col].dtype

                if dtype == np.float64:
                    dtype = np.int64
                elif dtype == np.float32:
                    dtype = np.int32
                elif dtype == np.float16:
                    dtype = np.int16

                df[col] = df[col].astype(dtype)

        data[dataset_id] = df

    cube = sparse_outer_cube.copy(uuid_prefix="sparse_outer_opt_cube")
    build_cube(data=data, store=module_store, cube=cube)
    return cube


@pytest.fixture(scope="module")
def massive_partitions_data():
    n = 17
    return {
        "seed": pd.DataFrame(
            {
                "x": np.arange(n),
                "y": np.arange(n),
                "z": np.arange(n),
                "p": np.arange(n),
                "q": np.arange(n),
                "v1": np.arange(n),
                "i1": np.arange(n),
            }
        ),
        "enrich_1": pd.DataFrame(
            {
                "x": np.arange(n),
                "y": np.arange(n),
                "z": np.arange(n),
                "p": np.arange(n),
                "q": np.arange(n),
                "v2": np.arange(n),
                "i2": np.arange(n),
            }
        ),
        "enrich_2": pd.DataFrame(
            {
                "y": np.arange(n),
                "z": np.arange(n),
                "p": np.arange(n),
                "q": np.arange(n),
                "v3": np.arange(n),
                "i3": np.arange(n),
            }
        ),
    }


@pytest.fixture(scope="module")
def massive_partitions_cube(module_store, massive_partitions_data):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="massive_partitions_cube",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(data=massive_partitions_data, store=module_store, cube=cube)
    return cube


@pytest.fixture(scope="module")
def fullrange_df():
    return (
        pd.DataFrame(
            data={
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v1": np.arange(16),
                "v2": np.arange(16),
                "v3": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
                "i1": np.arange(16),
                "i2": np.arange(16),
                "i3": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
            },
            columns=["i1", "i2", "i3", "p", "q", "v1", "v2", "v3", "x", "y", "z"],
        )
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def multipartition_df(fullrange_df):
    dfs = []
    for z in (0, 1):
        df = fullrange_df.copy()
        df["z"] = z
        dfs.append(df)
    return (
        pd.concat(dfs, ignore_index=True)
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def sparse_outer_df():
    return (
        pd.DataFrame(
            data={
                "x": [0, 1, 0],
                "y": [0, 0, 1],
                "z": 0,
                "p": [0, 1, 2],
                "q": 0,
                "v1": [0, 3, 7],
                "v2": [0, np.nan, 7],
                "v3": [0, 3, np.nan],
                "i1": [0, 3, 7],
                "i2": [0, np.nan, 7],
                "i3": [0, 3, np.nan],
            },
            columns=["i1", "i2", "i3", "p", "q", "v1", "v2", "v3", "x", "y", "z"],
        )
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def sparse_outer_opt_df(sparse_outer_df):
    df = sparse_outer_df.copy()
    df["x"] = df["x"].astype(np.int16)
    df["y"] = df["y"].astype(np.int32)
    df["z"] = df["z"].astype(np.int8)

    df["v1"] = df["v1"].astype(np.int8)
    df["i1"] = df["i1"].astype(np.int8)

    return df


@pytest.fixture(scope="module")
def massive_partitions_df():
    n = 17
    return (
        pd.DataFrame(
            data={
                "x": np.arange(n),
                "y": np.arange(n),
                "z": np.arange(n),
                "p": np.arange(n),
                "q": np.arange(n),
                "v1": np.arange(n),
                "v2": np.arange(n),
                "v3": np.arange(n),
                "i1": np.arange(n),
                "i2": np.arange(n),
                "i3": np.arange(n),
            },
            columns=["i1", "i2", "i3", "p", "q", "v1", "v2", "v3", "x", "y", "z"],
        )
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def updated_cube(module_store, fullrange_data):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="updated_cube",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={
            cube.seed_dataset: pd.DataFrame(
                {
                    "x": [0, 0, 1, 1, 2, 2],
                    "y": [0, 1, 0, 1, 0, 1],
                    "z": 0,
                    "p": [0, 0, 1, 1, 2, 2],
                    "q": 0,
                    "v1": np.arange(6),
                    "i1": np.arange(6),
                }
            ),
            "enrich": pd.DataFrame(
                {
                    "x": [0, 0, 1, 1, 2, 2],
                    "y": [0, 1, 0, 1, 0, 1],
                    "z": 0,
                    "p": [0, 0, 1, 1, 2, 2],
                    "q": 0,
                    "v2": np.arange(6),
                    "i2": np.arange(6),
                }
            ),
            "extra": pd.DataFrame(
                {
                    "y": [0, 1, 0, 1, 0, 1],
                    "z": 0,
                    "p": [0, 0, 1, 1, 2, 2],
                    "q": 0,
                    "v3": np.arange(6),
                    "i3": np.arange(6),
                }
            ),
        },
        store=module_store,
        cube=cube,
    )
    remove_partitions(
        cube=cube,
        store=module_store,
        ktk_cube_dataset_ids=["enrich"],
        conditions=C("p") >= 1,
    )
    append_to_cube(
        data={
            "enrich": pd.DataFrame(
                {
                    "x": [1, 1],
                    "y": [0, 1],
                    "z": 0,
                    "p": [1, 1],
                    "q": 0,
                    "v2": [7, 8],
                    "i2": [7, 8],
                }
            )
        },
        store=module_store,
        cube=cube,
    )
    return cube


@pytest.fixture(scope="module")
def updated_df():
    return (
        pd.DataFrame(
            data={
                "x": [0, 0, 1, 1, 2, 2],
                "y": [0, 1, 0, 1, 0, 1],
                "z": 0,
                "p": [0, 0, 1, 1, 2, 2],
                "q": 0,
                "v1": np.arange(6),
                "v2": [0, 1, 7, 8, np.nan, np.nan],
                "v3": np.arange(6),
                "i1": np.arange(6),
                "i2": [0, 1, 7, 8, np.nan, np.nan],
                "i3": np.arange(6),
            },
            columns=["i1", "i2", "i3", "p", "q", "v1", "v2", "v3", "x", "y", "z"],
        )
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def data_no_part():
    return {
        "seed": pd.DataFrame(
            {
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v1": np.arange(16),
                "i1": np.arange(16),
            }
        ),
        "enrich_dense": pd.DataFrame(
            {
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "v2": np.arange(16),
                "i2": np.arange(16),
            }
        ),
        "enrich_sparse": pd.DataFrame(
            {"y": [0, 1, 2, 3], "z": 0, "v3": np.arange(4), "i3": np.arange(4)}
        ),
    }


@pytest.fixture(scope="module")
def no_part_cube(module_store, data_no_part):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="data_no_part",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data=data_no_part,
        store=module_store,
        cube=cube,
        partition_on={"enrich_dense": [], "enrich_sparse": []},
    )
    return cube


@pytest.fixture(scope="module")
def other_part_cube(module_store, data_no_part):
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p", "q"],
        uuid_prefix="other_part_cube",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data=data_no_part,
        store=module_store,
        cube=cube,
        partition_on={"enrich_dense": ["i2"], "enrich_sparse": ["i3"]},
    )
    return cube


@pytest.fixture(scope="module")
def no_part_df():
    return (
        pd.DataFrame(
            data={
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "z": 0,
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v1": np.arange(16),
                "v2": np.arange(16),
                "v3": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "i1": np.arange(16),
                "i2": np.arange(16),
                "i3": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
            },
            columns=["i1", "i2", "i3", "p", "q", "v1", "v2", "v3", "x", "y", "z"],
        )
        .sort_values(["x", "y", "z", "p", "q"])
        .reset_index(drop=True)
    )


@pytest.fixture(
    params=[
        "fullrange",
        "multipartition",
        "sparse_outer",
        "sparse_outer_opt",
        "massive_partitions",
        "updated",
        "no_part",
        "other_part",
    ],
    scope="module",
)
def testset(request):
    return request.param


@pytest.fixture(scope="module")
def test_cube(
    testset,
    fullrange_cube,
    multipartition_cube,
    sparse_outer_cube,
    sparse_outer_opt_cube,
    massive_partitions_cube,
    updated_cube,
    no_part_cube,
    other_part_cube,
):
    if testset == "fullrange":
        return fullrange_cube
    elif testset == "multipartition":
        return multipartition_cube
    elif testset == "sparse_outer":
        return sparse_outer_cube
    elif testset == "sparse_outer_opt":
        return sparse_outer_opt_cube
    elif testset == "massive_partitions":
        return massive_partitions_cube
    elif testset == "updated":
        return updated_cube
    elif testset == "no_part":
        return no_part_cube
    elif testset == "other_part":
        return other_part_cube
    else:
        raise ValueError("Unknown param {}".format(testset))


@pytest.fixture(scope="module")
def test_df(
    testset,
    fullrange_df,
    multipartition_df,
    sparse_outer_df,
    sparse_outer_opt_df,
    massive_partitions_df,
    updated_df,
    no_part_df,
):
    if testset == "fullrange":
        return fullrange_df
    elif testset == "multipartition":
        return multipartition_df
    elif testset == "sparse_outer":
        return sparse_outer_df
    elif testset == "sparse_outer_opt":
        return sparse_outer_opt_df
    elif testset == "massive_partitions":
        return massive_partitions_df
    elif testset == "updated":
        return updated_df
    elif testset in ("no_part", "other_part"):
        return no_part_df
    else:
        raise ValueError("Unknown param {}".format(testset))


def test_simple_roundtrip(driver, function_store, function_store_rwro):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(data=df, cube=cube, store=function_store)
    result = driver(cube=cube, store=function_store_rwro)
    assert len(result) == 1
    df_actual = result[0]
    df_expected = df.reindex(columns=["p", "v", "x"])
    pdt.assert_frame_equal(df_actual, df_expected)


def test_complete(driver, module_store, test_cube, test_df):
    result = driver(cube=test_cube, store=module_store)
    assert len(result) == 1
    df_actual = result[0]
    pdt.assert_frame_equal(df_actual, test_df)


def apply_condition_unsafe(df, cond):
    # For the sparse_outer testset, the test_df has the wrong datatype because we cannot encode missing integer data in
    # pandas.
    #
    # The condition will not be applicable to the DF because the DF has floats while conditions have ints. We fix that
    # by modifying the the condition.
    #
    # In case there is no missing data because of the right conditions, kartothek will return integer data.
    # assert_frame_equal will then complain about this. So in case there is no missing data, let's recover the correct
    # dtype here.

    if not isinstance(cond, Conjunction):
        cond = Conjunction(cond)

    float_cols = {col for col in df.columns if df[col].dtype == float}

    # convert int to float conditions
    cond2 = Conjunction([])
    for col, conj in cond.split_by_column().items():
        if col in float_cols:
            parts = []
            for part in conj.conditions:
                if isinstance(part, IsInCondition):
                    part = IsInCondition(
                        column=part.column, value=tuple((float(v) for v in part.value))
                    )
                elif isinstance(part, InIntervalCondition):
                    part = InIntervalCondition(
                        column=part.column,
                        start=float(part.start),
                        stop=float(part.stop),
                    )
                else:
                    part = part.__class__(column=part.column, value=float(part.value))
                parts.append(part)
            conj = Conjunction(parts)
        cond2 &= conj

    # apply conditions
    df = cond2.filter_df(df).reset_index(drop=True)

    # convert float columns to int columns
    for col in df.columns:
        if df[col].notnull().all():
            dtype = df[col].dtype
            if dtype == np.float64:
                dtype = np.int64
            elif dtype == np.float32:
                dtype = np.int32
            elif dtype == np.float16:
                dtype = np.int16

            df[col] = df[col].astype(dtype)

    return df


@pytest.mark.parametrize(
    "cond",
    [
        C("v1") >= 7,
        C("v1") >= 10000,
        C("v2") >= 7,
        C("v3") >= 3,
        C("i1") >= 7,
        C("i1") >= 10000,
        C("i2") >= 7,
        C("i2") != 0,
        C("i3") >= 3,
        C("p") >= 1,
        C("q") >= 1,
        C("x") >= 1,
        C("y") >= 1,
        (C("x") == 3) & (C("y") == 3),
        (C("i1") > 0) & (C("i2") > 0),
        Conjunction([]),
    ],
)
def test_condition(driver, module_store, test_cube, test_df, cond):
    result = driver(cube=test_cube, store=module_store, conditions=cond)

    df_expected = apply_condition_unsafe(test_df, cond)

    if df_expected.empty:
        assert len(result) == 0
    else:
        assert len(result) == 1
        df_actual = result[0]
        pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("payload_columns", [["v1", "v2"], ["v2", "v3"], ["v3"]])
def test_select(driver, module_store, test_cube, test_df, payload_columns):
    result = driver(cube=test_cube, store=module_store, payload_columns=payload_columns)
    assert len(result) == 1
    df_actual = result[0]
    df_expected = test_df.loc[
        :, sorted(set(payload_columns) | {"x", "y", "z", "p", "q"})
    ]
    pdt.assert_frame_equal(df_actual, df_expected)


def test_filter_select(driver, module_store, test_cube, test_df):
    result = driver(
        cube=test_cube,
        store=module_store,
        payload_columns=["v1", "v2"],
        conditions=(C("i3") >= 3),  # completely unrelated to the payload
    )
    assert len(result) == 1
    df_actual = result[0]
    df_expected = test_df.loc[
        test_df["i3"] >= 3, ["p", "q", "v1", "v2", "x", "y", "z"]
    ].reset_index(drop=True)
    pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize(
    "partition_by",
    [["i1"], ["i2"], ["i3"], ["x"], ["y"], ["p"], ["q"], ["i1", "i2"], ["x", "y"]],
)
def test_partition_by(driver, module_store, test_cube, test_df, partition_by):
    dfs_actual = driver(cube=test_cube, store=module_store, partition_by=partition_by)

    dfs_expected = [
        df_g.reset_index(drop=True)
        for g, df_g in test_df.groupby(partition_by, sort=True)
    ]
    for df_expected in dfs_expected:
        for col in df_expected.columns:
            if df_expected[col].dtype == float:
                try:
                    df_expected[col] = df_expected[col].astype(int)
                except Exception:
                    pass

    assert len(dfs_actual) == len(dfs_expected)
    for df_actual, df_expected in zip(dfs_actual, dfs_expected):
        pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("dimension_columns", list(permutations(["x", "y", "z"])))
def test_sort(driver, module_store, test_cube, test_df, dimension_columns):
    result = driver(
        cube=test_cube, store=module_store, dimension_columns=dimension_columns
    )
    assert len(result) == 1
    df_actual = result[0]
    df_expected = test_df.sort_values(
        list(dimension_columns) + list(test_cube.partition_columns)
    ).reset_index(drop=True)
    pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("payload_columns", [["y", "z"], ["y", "z", "v3"]])
def test_projection(driver, module_store, test_cube, test_df, payload_columns):
    result = driver(
        cube=test_cube,
        store=module_store,
        dimension_columns=["y", "z"],
        payload_columns=payload_columns,
    )
    assert len(result) == 1
    df_actual = result[0]
    df_expected = (
        test_df.loc[:, sorted(set(payload_columns) | {"y", "z", "p", "q"})]
        .drop_duplicates()
        .sort_values(["y", "z", "p", "q"])
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(df_actual, df_expected)


def test_stresstest_index_select_row(driver, function_store):
    n_indices = 100
    n_rows = 1000

    data = {"x": np.arange(n_rows), "p": 0}
    for i in range(n_indices):
        data["i{}".format(i)] = np.arange(n_rows)
    df = pd.DataFrame(data)

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        index_columns=["i{}".format(i) for i in range(n_indices)],
    )

    build_cube(data=df, cube=cube, store=function_store)

    conditions = Conjunction([(C("i{}".format(i)) == 0) for i in range(n_indices)])

    result = driver(
        cube=cube,
        store=function_store,
        conditions=conditions,
        payload_columns=["p", "x"],
    )
    assert len(result) == 1
    df_actual = result[0]
    df_expected = df.loc[df["x"] == 0].reindex(columns=["p", "x"])
    pdt.assert_frame_equal(df_actual, df_expected)


def test_fail_missing_dimension_columns(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(cube=test_cube, store=module_store, dimension_columns=["x", "a", "b"])
    assert (
        "Following dimension columns were requested but are missing from the cube: a, b"
        in str(exc.value)
    )


def test_fail_empty_dimension_columns(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(cube=test_cube, store=module_store, dimension_columns=[])
    assert "Dimension columns cannot be empty." in str(exc.value)


def test_fail_missing_partition_by(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(cube=test_cube, store=module_store, partition_by=["foo"])
    assert (
        "Following partition-by columns were requested but are missing from the cube: foo"
        in str(exc.value)
    )


def test_fail_unindexed_partition_by(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(cube=test_cube, store=module_store, partition_by=["v1", "v2"])
    assert (
        "Following partition-by columns are not indexed and cannot be used: v1, v2"
        in str(exc.value)
    )


def test_fail_missing_condition_columns(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(
            cube=test_cube,
            store=module_store,
            conditions=(C("foo") == 1) & (C("bar") == 2),
        )
    assert (
        "Following condition columns are required but are missing from the cube: bar, foo"
        in str(exc.value)
    )


def test_fail_missing_payload_columns(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(cube=test_cube, store=module_store, payload_columns=["foo", "bar"])
    assert "Cannot find the following requested payload columns: bar, foo" in str(
        exc.value
    )


def test_fail_projection(driver, module_store, test_cube, test_df):
    with pytest.raises(ValueError) as exc:
        driver(
            cube=test_cube,
            store=module_store,
            dimension_columns=["y", "z"],
            payload_columns=["v1"],
        )
    assert (
        'Cannot project dataset "seed" with dimensionality [x, y, z] to [y, z] '
        "while keeping the following payload intact: v1" in str(exc.value)
    )


def test_fail_unstable_dimension_columns(driver, module_store, test_cube, test_df):
    with pytest.raises(TypeError) as exc:
        driver(cube=test_cube, store=module_store, dimension_columns={"x", "y"})
    assert "which has type set has an unstable iteration order" in str(exc.value)


def test_fail_unstable_partition_by(driver, module_store, test_cube, test_df):
    with pytest.raises(TypeError) as exc:
        driver(cube=test_cube, store=module_store, partition_by={"x", "y"})
    assert "which has type set has an unstable iteration order" in str(exc.value)


def test_wrong_condition_type(driver, function_store, driver_name):
    types = {
        "int": pd.Series([-1], dtype=np.int64),
        "uint": pd.Series([1], dtype=np.uint64),
        "float": pd.Series([1.3], dtype=np.float64),
        "bool": pd.Series([True], dtype=np.bool_),
        "str": pd.Series(["foo"], dtype=object),
    }
    cube = Cube(
        dimension_columns=["d_{}".format(t) for t in sorted(types.keys())],
        partition_columns=["p_{}".format(t) for t in sorted(types.keys())],
        uuid_prefix="typed_cube",
        index_columns=["i_{}".format(t) for t in sorted(types.keys())],
    )
    data = {
        "seed": pd.DataFrame(
            {
                "{}_{}".format(prefix, t): types[t]
                for t in sorted(types.keys())
                for prefix in ["d", "p", "v1"]
            }
        ),
        "enrich": pd.DataFrame(
            {
                "{}_{}".format(prefix, t): types[t]
                for t in sorted(types.keys())
                for prefix in ["d", "p", "i", "v2"]
            }
        ),
    }
    build_cube(data=data, store=function_store, cube=cube)

    df = pd.DataFrame(
        {
            "{}_{}".format(prefix, t): types[t]
            for t in sorted(types.keys())
            for prefix in ["d", "p", "i", "v1", "v2"]
        }
    )

    for col in df.columns:
        t1 = col.split("_")[1]

        for t2 in sorted(types.keys()):
            cond = C(col) == types[t2].values[0]

            if t1 == t2:
                result = driver(cube=cube, store=function_store, conditions=cond)
                assert len(result) == 1
                df_actual = result[0]
                df_expected = cond.filter_df(df).reset_index(drop=True)
                pdt.assert_frame_equal(df_actual, df_expected, check_like=True)
            else:
                with pytest.raises(TypeError) as exc:
                    driver(cube=cube, store=function_store, conditions=cond)
                assert "has wrong type" in str(exc.value)


def test_condition_on_null(driver, function_store):
    df = pd.DataFrame(
        {
            "x": pd.Series([0, 1, 2], dtype=np.int64),
            "p": pd.Series([0, 0, 1], dtype=np.int64),
            "v_f1": pd.Series([0, np.nan, 2], dtype=np.float64),
            "v_f2": pd.Series([0, 1, np.nan], dtype=np.float64),
            "v_f3": pd.Series([np.nan, np.nan, np.nan], dtype=np.float64),
            "v_s1": pd.Series(["a", None, "c"], dtype=object),
            "v_s2": pd.Series(["a", "b", None], dtype=object),
            "v_s3": pd.Series([None, None, None], dtype=object),
        }
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="nulled_cube",
        index_columns=[],
    )
    build_cube(data=df, store=function_store, cube=cube)

    for col in df.columns:
        # only iterate over the value columns (not the dimension / partition column):
        if not col.startswith("v"):
            continue

        # col_type will be either 'f' for float or 's' for string; see column
        # names above
        col_type = col.split("_")[1][0]
        if col_type == "f":
            value = 1.2
        elif col_type == "s":
            value = "foo"
        else:
            raise RuntimeError("unknown type")

        cond = C(col) == value

        df_expected = cond.filter_df(df).reset_index(drop=True)

        result = driver(cube=cube, store=function_store, conditions=cond)

        if df_expected.empty:
            assert len(result) == 0
        else:
            assert len(result) == 1
            df_actual = result[0]
            pdt.assert_frame_equal(df_actual, df_expected, check_like=True)


def test_fail_no_store_factory(driver, module_store, test_cube, skip_eager):
    store = module_store()
    with pytest.raises(TypeError) as exc:
        driver(cube=test_cube, store=store, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_delayed_index_build_partition_by(driver, function_store):
    df_seed = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1]})
    df_extend = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [0, 0, 0, 1]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="delayed_index_cube",
        index_columns=[],
    )
    build_cube(
        data={"seed": df_seed, "extend": df_extend}, store=function_store, cube=cube
    )

    build_dataset_indices(
        store=function_store,
        dataset_uuid=cube.ktk_dataset_uuid("extend"),
        columns=["v"],
    )

    results = driver(cube=cube, store=function_store, partition_by=["v"])
    assert len(results) == 2

    df_result1 = pd.DataFrame(
        data={"x": [0, 1, 2], "p": [0, 0, 1], "v": [0, 0, 0]}, columns=["p", "v", "x"]
    )
    df_result2 = pd.DataFrame(
        data={"x": [3], "p": [1], "v": [1]}, columns=["p", "v", "x"]
    )
    pdt.assert_frame_equal(results[0], df_result1)
    pdt.assert_frame_equal(results[1], df_result2)


def test_fail_blocksize_wrong_type(
    driver, module_store, test_cube, skip_eager, driver_name
):
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    with pytest.raises(TypeError, match="blocksize must be an integer but is str"):
        driver(cube=test_cube, store=module_store, blocksize="foo")


def test_fail_blocksize_negative(
    driver, module_store, test_cube, skip_eager, driver_name
):
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    with pytest.raises(ValueError, match="blocksize must be > 0 but is -1"):
        driver(cube=test_cube, store=module_store, blocksize=-1)


def test_fail_blocksize_zero(driver, module_store, test_cube, skip_eager, driver_name):
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    with pytest.raises(ValueError, match="blocksize must be > 0 but is 0"):
        driver(cube=test_cube, store=module_store, blocksize=0)


def test_delayed_index_build_correction_restriction(driver, function_store):
    """
    Ensure that adding extra indices for dimension columns does not mark other datasets as restrictive.
    """
    df_seed = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "p": [0, 0, 1, 1, 2, 2]})
    df_extend = pd.DataFrame({"x": [0, 1, 2], "p": [0, 0, 1], "v": [0, 1, 2]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="delayed_index_cube",
        index_columns=[],
    )
    build_cube(
        data={"seed": df_seed, "extend": df_extend}, store=function_store, cube=cube
    )

    build_dataset_indices(
        store=function_store,
        dataset_uuid=cube.ktk_dataset_uuid("extend"),
        columns=["x"],
    )

    results = driver(cube=cube, store=function_store, conditions=C("x") >= 0)
    assert len(results) == 1

    df_actual = results[0]
    df_expected = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "p": [0, 0, 1, 1, 2, 2],
            "v": [0, 1, 2, np.nan, np.nan, np.nan],
        },
        columns=["p", "v", "x"],
    )
    pdt.assert_frame_equal(df_actual, df_expected)


time_travel_stages_ops_df = [
    (
        partial(
            build_cube,
            data={
                "source": pd.DataFrame(
                    {
                        "x": [0, 1, 2, 3, 4, 5],
                        "p": [0, 0, 1, 1, 2, 2],
                        "v1": [0, 1, 2, 3, 4, 5],
                        "i1": [0, 1, 2, 3, 4, 5],
                    }
                ),
                "enrich": pd.DataFrame(
                    {
                        "x": [0, 1, 2, 3, 4, 5],
                        "p": [0, 0, 1, 1, 2, 2],
                        "v2": [0, 1, 2, 3, 4, 5],
                        "i2": [0, 1, 2, 3, 4, 5],
                    }
                ),
            },
        ),
        pd.DataFrame(
            data={
                "x": [0, 1, 2, 3, 4, 5],
                "p": [0, 0, 1, 1, 2, 2],
                "v1": [0, 1, 2, 3, 4, 5],
                "i1": [0, 1, 2, 3, 4, 5],
                "v2": [0, 1, 2, 3, 4, 5],
                "i2": [0, 1, 2, 3, 4, 5],
            },
            columns=["i1", "i2", "p", "v1", "v2", "x"],
        ),
    ),
    (
        partial(
            remove_partitions, ktk_cube_dataset_ids=["enrich"], conditions=C("p") > 0
        ),
        pd.DataFrame(
            data={
                "x": [0, 1, 2, 3, 4, 5],
                "p": [0, 0, 1, 1, 2, 2],
                "v1": [0, 1, 2, 3, 4, 5],
                "i1": [0, 1, 2, 3, 4, 5],
                "v2": [0, 1, np.nan, np.nan, np.nan, np.nan],
                "i2": [0, 1, np.nan, np.nan, np.nan, np.nan],
            },
            columns=["i1", "i2", "p", "v1", "v2", "x"],
        ),
    ),
    (
        partial(
            append_to_cube,
            data={"enrich": pd.DataFrame({"x": [2], "p": [1], "v2": [20], "i2": [20]})},
        ),
        pd.DataFrame(
            data={
                "x": [0, 1, 2, 3, 4, 5],
                "p": [0, 0, 1, 1, 2, 2],
                "v1": [0, 1, 2, 3, 4, 5],
                "i1": [0, 1, 2, 3, 4, 5],
                "v2": [0, 1, 20, np.nan, np.nan, np.nan],
                "i2": [0, 1, 20, np.nan, np.nan, np.nan],
            },
            columns=["i1", "i2", "p", "v1", "v2", "x"],
        ),
    ),
    (
        partial(
            append_to_cube,
            data={
                "source": pd.DataFrame(
                    {
                        "x": [4, 5, 6, 7],
                        "p": [2, 2, 3, 3],
                        "v1": [40, 50, 60, 70],
                        "i1": [40, 50, 60, 70],
                    }
                )
            },
        ),
        pd.DataFrame(
            data={
                "x": [0, 1, 2, 3, 4, 5, 6, 7],
                "p": [0, 0, 1, 1, 2, 2, 3, 3],
                "v1": [0, 1, 2, 3, 40, 50, 60, 70],
                "i1": [0, 1, 2, 3, 40, 50, 60, 70],
                "v2": [0, 1, 20, np.nan, np.nan, np.nan, np.nan, np.nan],
                "i2": [0, 1, 20, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            columns=["i1", "i2", "p", "v1", "v2", "x"],
        ),
    ),
]


def test_overlay_tricky(driver, function_store):
    cube = Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p", "q"],
        uuid_prefix="time_travel_cube_tricky",
        seed_dataset="source",
    )

    build_cube(
        data={
            cube.seed_dataset: pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                    "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    "v1": 1,
                }
            ),
            "no_part": pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                    "v2": 1,
                }
            ),
            "q": pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                    "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    "v3": 1,
                }
            ),
            "a": pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                    "a": [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                    "v4": 1,
                }
            ),
        },
        cube=cube,
        store=function_store,
        partition_on={"no_part": [], "q": ["q"], "a": ["a"]},
    )
    append_to_cube(
        data={
            cube.seed_dataset: pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 2, 3, 2, 3],
                    "y": [2, 2, 3, 3, 0, 0, 1, 1],
                    "p": [1, 1, 1, 1, 0, 0, 0, 0],
                    "q": [0, 0, 0, 0, 1, 1, 1, 1],
                    "v1": 2,
                }
            ),
            "no_part": pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                    "v2": 2,
                }
            ),
            "q": pd.DataFrame(
                {
                    "x": [0, 1, 0, 1, 0, 1, 0, 1],
                    "y": [0, 0, 1, 1, 2, 2, 3, 3],
                    "q": [0, 0, 0, 0, 0, 0, 0, 0],
                    "v3": 2,
                }
            ),
            "a": pd.DataFrame(
                {
                    "x": [1, 0, 1, 2, 3, 2, 3, 3],
                    "y": [0, 2, 2, 1, 1, 2, 2, 3],
                    "a": [1, 1, 1, 1, 1, 1, 1, 1],
                    "v4": 2,
                }
            ),
        },
        cube=cube,
        store=function_store,
    )

    df_expected = (
        pd.DataFrame(
            data={
                "x": [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
                "y": [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                "p": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "q": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "v1": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
                "v2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "v3": [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                "v4": [1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2],
                "a": [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
            },
            columns=["a", "p", "q", "v1", "v2", "v3", "v4", "x", "y"],
        )
        .sort_values(["x", "y", "p", "q"])
        .reset_index(drop=True)
    )

    result = driver(cube=cube, store=function_store)
    assert len(result) == 1
    df_actual = result[0]
    pdt.assert_frame_equal(df_actual, df_expected)


cond_types_simple = [
    EqualityCondition,
    LessEqualCondition,
    LessThanCondition,
    GreaterEqualCondition,
    GreaterThanCondition,
    InequalityCondition,
]

cond_types_all = cond_types_simple + [IsInCondition, InIntervalCondition]  # type:ignore


def _tuple_to_condition(t):
    col, cond_type, v1, v2, vset = t
    if issubclass(cond_type, tuple(cond_types_simple)):
        return cond_type(col, v1)
    elif cond_type == IsInCondition:
        return cond_type(col, vset)
    elif cond_type == InIntervalCondition:
        return cond_type(col, v1, v2)
    raise ValueError("Unknown condition type {}".format(cond_type))


st_columns = st.sampled_from(
    ["x", "y", "z", "p", "q", "i1", "i2", "i3", "v1", "v2", "v3"]
)
st_values = st.integers(min_value=-1, max_value=17)
st_cond_types = st.sampled_from(cond_types_all)
st_conditions = st.tuples(
    st_columns, st_cond_types, st_values, st_values, st.sets(st_values)
).map(_tuple_to_condition)


@given(
    conditions=st.lists(st_conditions).map(Conjunction),
    dimension_columns=st.permutations(["x", "y", "z"]),
    payload_columns=st.sets(st_columns),
)
@settings(deadline=timedelta(seconds=5))
def test_hypothesis(
    driver,
    driver_name,
    module_store,
    test_cube,
    test_df,
    dimension_columns,
    payload_columns,
    conditions,
):
    if driver_name != "eager":
        pytest.skip("only eager is fast enough")

    result = driver(
        cube=test_cube,
        store=module_store,
        dimension_columns=dimension_columns,
        payload_columns=payload_columns,
        conditions=conditions,
    )

    df_expected = (
        apply_condition_unsafe(test_df, conditions)
        .sort_values(dimension_columns + list(test_cube.partition_columns))
        .loc[:, sorted({"x", "y", "z", "p", "q"} | payload_columns)]
        .reset_index(drop=True)
    )

    if df_expected.empty:
        assert len(result) == 0
    else:
        assert len(result) == 1
        df_actual = result[0]
        pdt.assert_frame_equal(df_actual, df_expected)
