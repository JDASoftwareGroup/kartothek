# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.core.cube.cube import Cube
from kartothek.io.eager_cube import build_cube
from kartothek.utils.ktk_adapters import get_dataset_keys

__all__ = (
    "assert_same_keys",
    "built_cube",
    "cube",
    "df_enrich",
    "df_seed",
    "simple_cube_1",
    "simple_cube_2",
    "test_fail_blocksize_negative",
    "test_fail_blocksize_wrong_type",
    "test_fail_blocksize_zero",
    "test_fail_no_src_cube",
    "test_fail_no_src_cube_dataset",
    "test_fail_no_store_factory_src",
    "test_fail_no_store_factory_tgt",
    "test_fail_stores_identical_overwrite_false",
    "test_ignore_other",
    "test_invalid_partial_copy",
    "test_invalid_partial_copy1",
    "test_invalid_partial_copy2",
    "test_overwrite_fail",
    "test_overwrite_ok",
    "test_partial_copy_dataset_dict",
    "test_partial_copy_dataset_list",
    "test_read_only_source",
    "test_simple",
)


@pytest.fixture
def cube():
    return Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")


@pytest.fixture
def df_seed():
    return pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]})


@pytest.fixture
def df_enrich():
    return pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]})


@pytest.fixture
def built_cube(df_seed, df_enrich, cube, function_store):
    return build_cube(
        data={cube.seed_dataset: df_seed.copy(), "enrich": df_enrich.copy()},
        cube=cube,
        store=function_store,
    )


@pytest.fixture
def simple_cube_1(function_store, built_cube):
    return set(function_store().keys())


@pytest.fixture
def simple_cube_2(df_seed, df_enrich, cube, function_store2):
    build_cube(data={cube.seed_dataset: df_seed}, cube=cube, store=function_store2)
    return set(function_store2().keys())


def assert_same_keys(store1, store2, keys):
    k1 = set(store1().keys())
    k2 = set(store2().keys())
    assert keys.issubset(k1)
    assert keys.issubset(k2)

    for k in sorted(keys):
        b1 = store1().get(k)
        b2 = store1().get(k)
        assert b1 == b2


def test_simple(driver, function_store, function_store2, cube, simple_cube_1):
    driver(cube=cube, src_store=function_store, tgt_store=function_store2)
    assert_same_keys(function_store, function_store2, simple_cube_1)


def test_overwrite_fail(
    driver, function_store, function_store2, cube, simple_cube_1, simple_cube_2
):
    assert simple_cube_1 != simple_cube_2

    data_backup = {k: function_store2().get(k) for k in simple_cube_2}

    with pytest.raises(RuntimeError) as exc:
        driver(cube=cube, src_store=function_store, tgt_store=function_store2)
    assert (
        str(exc.value)
        == 'Dataset "cube++seed" exists in target store but overwrite was set to False'
    )

    # check everything kept untouched
    assert set(function_store2().keys()) == simple_cube_2
    for k in sorted(simple_cube_2):
        assert function_store2().get(k) == data_backup[k]


def test_overwrite_ok(
    driver, function_store, function_store2, cube, simple_cube_1, simple_cube_2
):
    driver(
        cube=cube, src_store=function_store, tgt_store=function_store2, overwrite=True
    )
    assert_same_keys(function_store, function_store2, simple_cube_1)


@pytest.mark.parametrize("overwrite", [False, True])
def test_fail_stores_identical_overwrite_false(
    driver, function_store, cube, built_cube, overwrite
):
    with pytest.raises(ValueError) as exc:
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store,
            overwrite=overwrite,
        )
    assert str(exc.value) == "Stores are identical but should not be."


def test_ignore_other(driver, function_store, function_store2):
    dfs = []
    cubes = []
    for i in range(3):
        dfs.append(
            pd.DataFrame(
                {
                    "x{}".format(i): [0, 1, 2, 3],
                    "p": [0, 0, 1, 1],
                    "v{}".format(i): [10, 11, 12, 13],
                }
            )
        )

        cubes.append(
            Cube(
                dimension_columns=["x{}".format(i)],
                partition_columns=["p"],
                uuid_prefix="cube{}".format(i),
            )
        )

    build_cube(data=dfs[0], cube=cubes[0], store=function_store)
    build_cube(data=dfs[1], cube=cubes[1], store=function_store)
    build_cube(data=dfs[2], cube=cubes[2], store=function_store2)

    keys_in_1 = set(function_store().keys())
    keys_in_2 = set(function_store2().keys())
    data_backup1 = {k: function_store().get(k) for k in keys_in_1}
    data_backup2 = {k: function_store2().get(k) for k in keys_in_2}

    driver(cube=cubes[1], src_store=function_store, tgt_store=function_store2)

    # store 1 is untouched
    assert set(function_store().keys()) == keys_in_1
    for k in sorted(keys_in_1):
        assert function_store().get(k) == data_backup1[k]

    # store 2 is partly untouched
    for k in sorted(keys_in_2):
        assert function_store2().get(k) == data_backup2[k]

    # test new keys
    keys_new = set(function_store2().keys()) - keys_in_2
    assert_same_keys(function_store, function_store2, keys_new)


def test_invalid_partial_copy1(
    df_seed, df_enrich, cube, function_store, function_store2, simple_cube_2, driver
):
    # build a cube that would be incompatible w/ simple_cube_2
    df_seed = df_seed.copy()
    df_enrich = df_enrich.copy()

    df_seed["x"] = df_seed["x"].astype(str)
    df_enrich["x"] = df_enrich["x"].astype(str)
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store,
    )

    keys = set(function_store().keys())

    # now copy simple_cube_2 over existing cube.
    # this only copies the seed table since simple_cube_2 does not have an enrich table.
    # it should fail because X is incompatible
    with pytest.raises(ValueError) as exc:
        driver(
            cube=cube,
            src_store=function_store2,
            tgt_store=function_store,
            overwrite=True,
        )
    assert 'Found incompatible entries for column "x"' in str(exc.value)
    assert keys == set(function_store().keys())


def test_invalid_partial_copy2(
    df_seed, df_enrich, cube, function_store, function_store2, simple_cube_1, driver
):
    # build a cube that would be incompatible w/ simple_cube_1
    df_seed = df_seed.copy()
    df_enrich = df_enrich.copy()

    df_seed["x"] = df_seed["x"].astype(str)
    df_enrich["x"] = df_enrich["x"].astype(str)
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich2": df_enrich},
        cube=cube,
        store=function_store2,
    )

    keys = set(function_store2().keys())

    # now copy simple_cube_1 over existing cube.
    # this only copies the seed and enrich table since simple_cube_1 does not have an enrich2 table.
    # it should fail because X is incompatible.
    with pytest.raises(ValueError) as exc:
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store2,
            overwrite=True,
        )
    assert "Found columns present in multiple datasets" in str(exc.value)
    assert keys == set(function_store2().keys())


def test_partial_copy_dataset_list(
    driver, function_store, function_store2, cube, built_cube
):
    driver(
        cube=cube,
        src_store=function_store,
        tgt_store=function_store2,
        datasets=["seed", "enrich"],
    )
    all_datasets = discover_datasets_unchecked(
        uuid_prefix=cube.uuid_prefix,
        store=function_store,
        filter_ktk_cube_dataset_ids=["seed", "enrich"],
    )
    copied_ds_keys = set()
    copied_ds_keys |= get_dataset_keys(all_datasets["seed"])
    copied_ds_keys |= get_dataset_keys(all_datasets["enrich"])
    tgt_store_keys = set(function_store2().keys())
    assert copied_ds_keys == tgt_store_keys


def test_partial_copy_dataset_dict(
    driver, function_store, function_store2, cube, built_cube
):
    driver(
        cube=cube,
        src_store=function_store,
        tgt_store=function_store2,
        datasets={"seed": built_cube["seed"], "enrich": built_cube["enrich"]},
    )
    all_datasets = discover_datasets_unchecked(
        uuid_prefix=cube.uuid_prefix,
        store=function_store,
        filter_ktk_cube_dataset_ids=["seed", "enrich"],
    )
    copied_ds_keys = set()
    copied_ds_keys |= get_dataset_keys(all_datasets["seed"])
    copied_ds_keys |= get_dataset_keys(all_datasets["enrich"])
    tgt_store_keys = set(function_store2().keys())
    assert copied_ds_keys == tgt_store_keys


def test_invalid_partial_copy(
    driver, df_seed, df_enrich, function_store, function_store2, cube, built_cube
):
    # build a cube that would be incompatible with cube in function_store
    df_seed = df_seed.copy()
    df_enrich = df_enrich.copy()
    df_seed["x"] = df_seed["x"].astype(str)
    df_enrich["x"] = df_enrich["x"].astype(str)
    build_cube(
        data={cube.seed_dataset: df_seed, "enrich": df_enrich},
        cube=cube,
        store=function_store2,
    )
    tgt_store_key_before = set(function_store2().keys())
    with pytest.raises(ValueError) as exc:
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store2,
            overwrite=True,
            datasets=["enrich"],
        )
    assert 'Found incompatible entries for column "x"' in str(exc.value)
    assert tgt_store_key_before == set(function_store2().keys())


def test_fail_no_store_factory_src(
    driver, function_store, function_store2, cube, skip_eager
):
    store = function_store()
    with pytest.raises(TypeError) as exc:
        driver(cube=cube, src_store=store, tgt_store=function_store2, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_fail_no_store_factory_tgt(
    driver, function_store, function_store2, cube, skip_eager
):
    store = function_store2()
    with pytest.raises(TypeError) as exc:
        driver(cube=cube, src_store=function_store, tgt_store=store, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_fail_no_src_cube(cube, function_store, function_store2, driver):
    with pytest.raises(RuntimeError) as exc:
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store2,
            overwrite=False,
        )
    assert "not found" in str(exc.value)


def test_fail_no_src_cube_dataset(
    cube, built_cube, function_store, function_store2, driver
):
    with pytest.raises(RuntimeError) as exc:
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store2,
            overwrite=False,
            datasets=["non_existing"],
        )
    assert "non_existing" in str(exc.value)


def test_read_only_source(
    driver, function_store_ro, function_store2, cube, simple_cube_1
):
    driver(cube=cube, src_store=function_store_ro, tgt_store=function_store2)
    assert_same_keys(function_store_ro, function_store2, simple_cube_1)


def test_fail_blocksize_wrong_type(
    driver, function_store, function_store2, cube, simple_cube_1, skip_eager
):
    with pytest.raises(TypeError, match="blocksize must be an integer but is str"):
        driver(
            cube=cube,
            src_store=function_store,
            tgt_store=function_store2,
            blocksize="foo",
        )


def test_fail_blocksize_negative(
    driver, function_store, function_store2, cube, simple_cube_1, skip_eager
):
    with pytest.raises(ValueError, match="blocksize must be > 0 but is -1"):
        driver(
            cube=cube, src_store=function_store, tgt_store=function_store2, blocksize=-1
        )


def test_fail_blocksize_zero(
    driver, function_store, function_store2, cube, simple_cube_1, skip_eager
):
    with pytest.raises(ValueError, match="blocksize must be > 0 but is 0"):
        driver(
            cube=cube, src_store=function_store, tgt_store=function_store2, blocksize=0
        )
