# -*- coding: utf-8 -*-
import dask.bag as db
import pandas as pd
import pytest

from kartothek.core.cube.cube import Cube
from kartothek.io.dask.bag_cube import build_cube_from_bag
from kartothek.io.eager_cube import build_cube

__all__ = (
    "test_fail_blocksize_negative",
    "test_fail_blocksize_wrong_type",
    "test_fail_blocksize_zero",
    "test_fail_no_store_factory",
    "test_multifile",
    "test_simple",
)


def test_simple(driver, function_store, function_store_rwro):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "foo": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
    )
    result = driver(cube=cube, store=function_store_rwro)

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}
    stats_seed = result[cube.seed_dataset]

    assert stats_seed["partitions"] == 2
    assert stats_seed["files"] == 2
    assert stats_seed["rows"] == 4
    assert stats_seed["blobsize"] > 0

    stats_enrich = result["enrich"]
    assert stats_enrich["partitions"] == stats_seed["partitions"]
    assert stats_enrich["files"] == stats_seed["files"]
    assert stats_enrich["rows"] == stats_seed["rows"]
    assert stats_enrich["blobsize"] != stats_seed["blobsize"]


def test_multifile(driver, function_store):
    dfs = [pd.DataFrame({"x": [i], "p": [0], "v1": [10]}) for i in range(2)]
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube_from_bag(
        data=db.from_sequence(dfs, partition_size=1), cube=cube, store=function_store
    ).compute()

    result = driver(cube=cube, store=function_store)

    assert set(result.keys()) == {cube.seed_dataset}
    stats_seed = result[cube.seed_dataset]
    assert stats_seed["partitions"] == 1
    assert stats_seed["files"] == 2
    assert stats_seed["rows"] == 2
    assert stats_seed["blobsize"] > 0


def test_fail_no_store_factory(driver, function_store, skip_eager):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(data=df_seed, cube=cube, store=function_store)
    store = function_store()
    with pytest.raises(TypeError) as exc:
        driver(cube=cube, store=store, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_fail_blocksize_wrong_type(driver, function_store, skip_eager):
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    with pytest.raises(TypeError, match="blocksize must be an integer but is str"):
        driver(cube=cube, store=function_store, blocksize="foo")


def test_fail_blocksize_negative(driver, function_store, skip_eager):
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    with pytest.raises(ValueError, match="blocksize must be > 0 but is -1"):
        driver(cube=cube, store=function_store, blocksize=-1)


def test_fail_blocksize_zero(driver, function_store, skip_eager):
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    with pytest.raises(ValueError, match="blocksize must be > 0 but is 0"):
        driver(cube=cube, store=function_store, blocksize=0)
