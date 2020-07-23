# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.core.cube.cube import Cube
from kartothek.io.eager_cube import build_cube
from kartothek.utils.ktk_adapters import get_dataset_keys

__all__ = (
    "test_delete_twice",
    "test_fail_blocksize_negative",
    "test_fail_blocksize_wrong_type",
    "test_fail_blocksize_zero",
    "test_fail_no_store_factory",
    "test_keep_garbage_due_to_no_listing",
    "test_keep_other",
    "test_partial_delete",
    "test_simple",
)


def test_simple(driver, function_store):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
    )
    driver(cube=cube, store=function_store)

    assert set(function_store().keys()) == set()


def test_keep_other(driver, function_store):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube1 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube1")
    cube2 = cube1.copy(uuid_prefix="cube2")

    build_cube(data=df, cube=cube1, store=function_store)
    keys = set(function_store().keys())

    build_cube(data=df, cube=cube2, store=function_store)

    driver(cube=cube2, store=function_store)

    assert set(function_store().keys()) == keys


def test_keep_garbage_due_to_no_listing(driver, function_store):
    df1 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    df2 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [2, 2, 3, 3], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    # test build DF1 to see which keys are created
    build_cube(data=df1, cube=cube, store=function_store)
    keys1 = set(function_store().keys())

    # wipe
    for k in list(function_store().keys()):
        function_store().delete(k)

    # test build DF2 to see which keys are created
    build_cube(data=df2, cube=cube, store=function_store)
    keys2 = set(function_store().keys())

    # wipe again
    for k in list(function_store().keys()):
        function_store().delete(k)

    # some keys are obviosly present everytime (like central metadata and
    # common metadata)
    keys_common = keys1 & keys2

    # build DF1 and overwrite w/ DF2
    build_cube(data=df1, cube=cube, store=function_store)
    keys3 = set(function_store().keys())

    build_cube(data=df2, cube=cube, store=function_store, overwrite=True)

    # now some keys if DF1 must be leftovers/gargabe that cannot be deleted w/o listing the entire store (which would
    # be too expensive)
    gargabe = keys3 - keys_common
    assert len(gargabe) > 0

    driver(cube=cube, store=function_store)

    assert set(function_store().keys()) == gargabe


def test_delete_twice(driver, function_store):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(data=df, cube=cube, store=function_store)
    driver(cube=cube, store=function_store)
    driver(cube=cube, store=function_store)

    assert set(function_store().keys()) == set()


def test_partial_delete(driver, function_store):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]}
    )
    df_1 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "a": [20, 21, 22, 23]})
    df_2 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "b": [20, 21, 22, 23]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    datasets = build_cube(
        data={cube.seed_dataset: df_seed, "enrich-1": df_1, "enrich-2": df_2},
        cube=cube,
        store=function_store,
    )
    enrich_1_keys = get_dataset_keys(
        discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=function_store,
            filter_ktk_cube_dataset_ids=["enrich-1"],
        )["enrich-1"]
    )
    enrich_2_keys = get_dataset_keys(
        discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=function_store,
            filter_ktk_cube_dataset_ids=["enrich-2"],
        )["enrich-2"]
    )
    all_keys = set(function_store().keys())
    driver(cube=cube, store=function_store, datasets=["enrich-1"])
    assert set(function_store().keys()) == all_keys - enrich_1_keys

    driver(cube=cube, store=function_store, datasets={"enrich-2": datasets["enrich-2"]})
    assert set(function_store().keys()) == all_keys - enrich_1_keys - enrich_2_keys


def test_fail_no_store_factory(driver, function_store, skip_eager):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
    )
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
