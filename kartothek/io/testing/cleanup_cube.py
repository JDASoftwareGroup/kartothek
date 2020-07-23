# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from kartothek.core.cube.constants import KTK_CUBE_UUID_SEPERATOR
from kartothek.core.cube.cube import Cube
from kartothek.io.eager_cube import build_cube, copy_cube

__all__ = (
    "test_additional_files",
    "test_delete_by_correct_uuid",
    "test_fail_blocksize_negative",
    "test_fail_blocksize_wrong_type",
    "test_fail_blocksize_zero",
    "test_fails_no_store_factory",
    "test_missing_cube_files",
    "test_missing_metadata",
    "test_missing_seed_dataset",
    "test_noop",
    "test_overwrite_check_with_copy",
)


def test_delete_by_correct_uuid(driver, function_store):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]}
    )
    cube_foo = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="foo")
    build_cube(
        data={cube_foo.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube_foo,
        store=function_store,
    )

    cube_foo_bar = Cube(
        dimension_columns=["x"], partition_columns=["p"], uuid_prefix="foo_bar"
    )
    build_cube(
        data={cube_foo_bar.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube_foo_bar,
        store=function_store,
    )
    store = function_store()
    foo_bar_keys = {k for k in store.keys() if "foo_bar" in k}
    store.delete("foo++seed.by-dataset-metadata.json")
    store.delete("foo++enrich.by-dataset-metadata.json")

    driver(cube=cube_foo, store=function_store)
    assert foo_bar_keys == set(store.keys())


def test_missing_seed_dataset(driver, function_store):
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
    seed_keys = {k for k in store.keys() if "cube++seed" in k and "/" in k}
    enrich_keys = {k for k in store.keys() if "cube++enrich" in k}

    for k in seed_keys:
        store.delete(k)

    driver(cube=cube, store=function_store)

    assert enrich_keys == set(store.keys())


def test_missing_cube_files(driver, function_store):
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
    enrich_keys = {k for k in store.keys() if "cube++enrich" in k and "/" in k}
    for k in enrich_keys:
        store.delete(k)

    driver(cube=cube, store=function_store)

    assert "cube++enrich.by-dataset-metadata.json" not in store.keys()


def test_missing_metadata(driver, function_store):
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
    enrich_keys = {k for k in store.keys() if "cube++enrich" in k}

    store.delete("cube++enrich.by-dataset-metadata.json")

    driver(cube=cube, store=function_store)

    assert not enrich_keys.intersection(store.keys())


def test_noop(driver, function_store):
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

    keys = set(function_store().keys())

    driver(cube=cube, store=function_store)

    assert set(function_store().keys()) == keys


def test_overwrite_check_with_copy(driver, function_store, function_store2):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    # build twice
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
    )
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        overwrite=True,
    )

    # copy to another store to detect keys
    copy_cube(cube=cube, src_store=function_store, tgt_store=function_store2)
    keys = set(function_store2().keys())

    assert set(function_store().keys()) != keys
    driver(cube=cube, store=function_store)
    assert set(function_store().keys()) == keys


def test_additional_files(driver, function_store):
    df_seed = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    build_cube(data=df_seed, cube=cube, store=function_store)

    key_in_ds = cube.ktk_dataset_uuid(cube.seed_dataset) + "/foo"
    key_with_ds_prefix = cube.ktk_dataset_uuid(cube.seed_dataset) + ".foo"
    key_with_cube_prefix = cube.uuid_prefix + ".foo"
    key_with_cube_prefix_separator = cube.uuid_prefix + KTK_CUBE_UUID_SEPERATOR + ".foo"

    function_store().put(key_in_ds, b"")
    function_store().put(key_with_ds_prefix, b"")
    function_store().put(key_with_cube_prefix, b"")
    function_store().put(key_with_cube_prefix_separator, b"")

    driver(cube=cube, store=function_store)
    assert key_in_ds not in set(function_store().keys())
    assert key_with_ds_prefix not in set(function_store().keys())
    assert key_with_cube_prefix in set(function_store().keys())
    assert key_with_cube_prefix_separator not in set(function_store().keys())


def test_fails_no_store_factory(driver, function_store, skip_eager):
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

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
