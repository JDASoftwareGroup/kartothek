import pandas as pd
import pytest

from kartothek.api.discover import discover_datasets
from kartothek.core.cube.conditions import C
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.eager_cube import build_cube, remove_partitions


@pytest.fixture
def driver(driver_name):
    if driver_name in ("dask_bag_bs1", "dask_bag_bs3"):
        pytest.skip("not implemented yet")
    elif driver_name == "dask_dataframe":
        pytest.skip("not supported for dask.dataframe")
    elif driver_name == "eager":
        return remove_partitions
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))


def _get_cube(function_store, with_partition_on):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "q": 0, "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "q": 0, "v2": [10, 11, 12, 13]}
    )
    if with_partition_on:
        df_enrich.drop(columns=["p", "q"], inplace=True)

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p", "q"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        metadata={"source": {"userkey1": "value1"}},
        partition_on={"enrich": []} if with_partition_on else None,
    )
    return cube


@pytest.fixture(params=[True, False])
def with_partition_on(request):
    return request.param


@pytest.fixture
def existing_cube(function_store, with_partition_on):
    return _get_cube(function_store, with_partition_on=with_partition_on)


def test_all(driver, function_store, existing_cube):
    result = driver(cube=existing_cube, store=function_store)

    assert set(result.keys()) == {"source", "enrich"}

    ds_source = result["source"]
    ds_enrich = result["enrich"]

    assert len(ds_source.partitions) == 0
    assert len(ds_enrich.partitions) == 0

    discover_datasets(existing_cube, function_store)


def test_conditions(driver, function_store, existing_cube):
    parts_source1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("source"), function_store()
        ).partitions
    )
    parts_enrich1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("enrich"), function_store()
        ).partitions
    )

    parts_source_to_delete = {part for part in parts_source1 if "p=0" not in part}

    result = driver(
        cube=existing_cube,
        store=function_store,
        ktk_cube_dataset_ids=["source"],
        conditions=C("p") > 0,
    )

    assert set(result.keys()) == {"source", "enrich"}

    ds_source = result["source"]
    ds_enrich = result["enrich"]

    parts_source2 = set(ds_source.partitions)
    parts_enrich2 = set(ds_enrich.partitions)

    assert parts_enrich1 == parts_enrich2
    assert parts_source1 - parts_source_to_delete == parts_source2


def test_remove_nonmatching_condition(driver, function_store, existing_cube):
    parts_source_before = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("source"), function_store()
        ).partitions
    )
    result = driver(
        cube=existing_cube,
        store=function_store,
        ktk_cube_dataset_ids=["source"],
        conditions=C("p") > 10000,
    )
    parts_source_after = set(result["source"].partitions)
    assert parts_source_before == parts_source_after


def test_fail_wrong_condition(driver, function_store, existing_cube):
    with pytest.raises(
        ValueError,
        match="Can only remove partitions with conditions concerning cubes physical partition columns.",
    ):
        driver(
            cube=existing_cube,
            store=function_store,
            ktk_cube_dataset_ids=["source"],
            conditions=C("v1") >= 0,
        )


def test_fail_id_not_str(driver, function_store, existing_cube):
    with pytest.raises(TypeError, match="Object of type int is not a string: 1"):
        driver(cube=existing_cube, store=function_store, ktk_cube_dataset_ids=[1])


def test_fail_ids_unknown(driver, function_store, existing_cube):
    with pytest.raises(ValueError, match="Unknown ktk_cube_dataset_ids: bar, foo"):
        driver(
            cube=existing_cube,
            store=function_store,
            ktk_cube_dataset_ids=["foo", "bar", "source"],
        )
