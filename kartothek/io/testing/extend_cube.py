import pandas as pd
import pytest

from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
from kartothek.io.eager_cube import build_cube
from kartothek.io_components.cube.write import MultiTableCommitAborted
from kartothek.io_components.metapartition import SINGLE_TABLE

__all__ = (
    "existing_cube",
    "test_fail_all_empty",
    "test_fail_no_store_factory",
    "test_fail_not_a_df",
    "test_fail_wrong_dataset_ids",
    "test_fails_incompatible_dtypes",
    "test_fails_metadata_nested_wrong_type",
    "test_fails_metadata_unknown_id",
    "test_fails_metadata_wrong_type",
    "test_fails_overlapping_payload_enrich",
    "test_fails_overlapping_payload_overwrite",
    "test_fails_overlapping_payload_partial",
    "test_fails_overlapping_payload_seed",
    "test_fails_seed_dataset",
    "test_overwrite_move_columns",
    "test_overwrite_single",
    "test_simple",
)


@pytest.fixture
def existing_cube(function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={"source": df_source, "enrich": df_enrich}, cube=cube, store=function_store
    )
    return cube


def test_simple(driver, function_store, existing_cube):
    """
    Simple integration test w/ single extra dataset.
    """
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v3": [10, 11, 12, 13],
            "i3": [100, 101, 102, 103],
        }
    )
    result = driver(data={"extra": df}, cube=existing_cube, store=function_store)

    assert set(result.keys()) == {"extra"}

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert ds.uuid == existing_cube.ktk_dataset_uuid("extra")
    assert len(ds.partitions) == 2

    assert set(ds.indices.keys()) == {"p", "i3"}
    assert isinstance(ds.indices["p"], PartitionIndex)
    assert isinstance(ds.indices["i3"], ExplicitSecondaryIndex)

    assert set(ds.table_meta) == {SINGLE_TABLE}


def test_fails_incompatible_dtypes(driver, function_store, existing_cube):
    """
    Should also cross check w/ seed dataset.
    """
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "p": [0, 0, 1, 1],
            "v3": [10, 11, 12, 13],
            "i3": [100, 101, 102, 103],
        }
    )
    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(data={"extra": df}, cube=existing_cube, store=function_store)
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert 'Found incompatible entries for column "x"' in str(cause)
    assert not DatasetMetadata.exists(
        existing_cube.ktk_dataset_uuid("extra"), function_store()
    )


def test_fails_seed_dataset(driver, function_store, existing_cube):
    """
    Users cannot overwrite seed dataset since it is used for consisteny checks.
    """
    pre_keys = set(function_store().keys())
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]})
    with pytest.raises(ValueError) as exc:
        driver(
            data={existing_cube.seed_dataset: df},
            cube=existing_cube,
            store=function_store,
        )
    assert 'Seed data ("source") cannot be written during extension.' in str(exc.value)

    post_keys = set(function_store().keys())
    assert pre_keys == post_keys


def test_fails_overlapping_payload_seed(driver, function_store, existing_cube):
    """
    Forbidden by spec, results in problems during query.
    """
    pre_keys = set(function_store().keys())
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]})
    with pytest.raises(ValueError) as exc:
        driver(data={"extra": df}, cube=existing_cube, store=function_store)
    assert 'Payload written in "extra" is already present in cube: v1' in str(exc.value)
    assert not DatasetMetadata.exists(
        existing_cube.ktk_dataset_uuid("extra"), function_store()
    )

    post_keys = set(function_store().keys())
    assert pre_keys == post_keys


def test_fails_overlapping_payload_enrich(driver, function_store, existing_cube):
    """
    Forbidden by spec, results in problems during query.
    """
    pre_keys = set(function_store().keys())
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]})
    with pytest.raises(ValueError) as exc:
        driver(data={"extra": df}, cube=existing_cube, store=function_store)
    assert 'Payload written in "extra" is already present in cube: v2' in str(exc.value)
    assert not DatasetMetadata.exists(
        existing_cube.ktk_dataset_uuid("extra"), function_store()
    )

    post_keys = set(function_store().keys())
    assert pre_keys == post_keys


def test_fails_overlapping_payload_partial(driver, function_store, existing_cube):
    """
    Forbidden by spec, results in problems during query.
    """
    pre_keys = set(function_store().keys())
    df1 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [10, 11, 12, 13]})
    df2 = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]})
    with pytest.raises(ValueError) as exc:
        driver(
            data={"extra1": df1, "extra2": df2},
            cube=existing_cube,
            store=function_store,
        )
    assert 'Payload written in "extra1" is already present in cube: v2' in str(
        exc.value
    )

    assert not DatasetMetadata.exists(
        existing_cube.ktk_dataset_uuid("extra1"), function_store()
    )
    # extra2 might exist, depending on the compute graph

    # extra2 keys might be present, only look that extra1 is absent
    post_keys = set(function_store().keys())
    extra_keys = post_keys - pre_keys
    extra1_keys = {k for k in extra_keys if "extra1" in k}
    assert extra1_keys == set()


def test_fails_overlapping_payload_overwrite(driver, function_store, existing_cube):
    """
    Forbidden by spec, results in problems during query.
    """
    pre_keys = set(function_store().keys())
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]})
    with pytest.raises(ValueError) as exc:
        driver(
            data={"enrich": df},
            cube=existing_cube,
            store=function_store,
            overwrite=True,
        )
    assert 'Payload written in "enrich" is already present in cube: v1' in str(
        exc.value
    )

    post_keys = set(function_store().keys())
    assert pre_keys == post_keys


def test_overwrite_single(driver, function_store, existing_cube):
    """
    Simple overwrite of the enrich dataset.
    """
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]})

    # does not work w/o explicit flag
    keys = set(function_store().keys())
    with pytest.raises(RuntimeError) as exc:
        driver(data={"enrich": df}, cube=existing_cube, store=function_store)
    assert "already exists" in str(exc.value)
    assert set(function_store().keys()) == keys

    # but works with flag
    result = driver(
        data={"enrich": df}, cube=existing_cube, store=function_store, overwrite=True
    )

    assert set(result.keys()) == {"enrich"}

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert ds.uuid == existing_cube.ktk_dataset_uuid("enrich")
    assert len(ds.partitions) == 2


def test_overwrite_move_columns(driver, function_store, existing_cube):
    """
    Move columns v1 and i1 from enrich to extra.
    """
    df_enrich = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v3": [10, 11, 12, 13],
            "i3": [100, 101, 102, 103],
        }
    )
    df_extra = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v2": [10, 11, 12, 13],
            "i2": [100, 101, 102, 103],
        }
    )

    result = driver(
        data={"enrich": df_enrich, "extra": df_extra},
        cube=existing_cube,
        store=function_store,
        overwrite=True,
    )

    assert set(result.keys()) == {"enrich", "extra"}

    ds_enrich = result["enrich"].load_all_indices(function_store())
    ds_extra = result["extra"].load_all_indices(function_store())

    assert set(ds_enrich.indices.keys()) == {"p", "i3"}
    assert isinstance(ds_enrich.indices["p"], PartitionIndex)
    assert isinstance(ds_enrich.indices["i3"], ExplicitSecondaryIndex)

    assert set(ds_extra.indices.keys()) == {"p", "i2"}
    assert isinstance(ds_extra.indices["p"], PartitionIndex)
    assert isinstance(ds_extra.indices["i2"], ExplicitSecondaryIndex)


def test_fail_all_empty(driver, function_store, existing_cube):
    """
    Might happen due to DB-based filters.
    """
    df = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]}
    ).loc[[]]

    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(data={"extra": df}, cube=existing_cube, store=function_store)
    exc = exc_info.value.__cause__
    assert isinstance(exc, ValueError)
    assert "Cannot write empty datasets: extra" in str(exc)
    assert not DatasetMetadata.exists(
        existing_cube.ktk_dataset_uuid("extra"), function_store()
    )


def test_fail_not_a_df(driver, function_store, existing_cube):
    """
    Pass some weird objects in.
    """
    with pytest.raises(TypeError) as exc:
        driver(
            data={"extra": pd.Series(range(10))},
            cube=existing_cube,
            store=function_store,
        )
    assert (
        'Provided DataFrame is not a pandas.DataFrame or None, but is a "Series"'
        in str(exc.value)
    )


def test_fail_wrong_dataset_ids(
    driver, function_store, existing_cube, skip_eager, driver_name
):
    if driver_name == "dask_dataframe":
        pytest.skip("not an interface for dask.dataframe")

    df_extra = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    with pytest.raises(ValueError) as exc:
        driver(
            data={"extra": df_extra},
            cube=existing_cube,
            store=function_store,
            ktk_cube_dataset_ids=["other"],
        )

    assert (
        'Ktk_cube Dataset ID "extra" is present during pipeline execution '
        "but was not specified in ktk_cube_dataset_ids (other)." in str(exc.value)
    )


def test_fail_no_store_factory(driver, function_store, existing_cube, skip_eager):
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v3": [10, 11, 12, 13],
            "i3": [100, 101, 102, 103],
        }
    )
    store = function_store()
    with pytest.raises(TypeError) as exc:
        driver(data={"extra": df}, cube=existing_cube, store=store, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_fails_metadata_wrong_type(driver, function_store, existing_cube):
    df_extra = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]}
    )
    with pytest.raises(
        TypeError, match="Provided metadata should be a dict but is int"
    ):
        driver(
            data={"extra": df_extra},
            cube=existing_cube,
            store=function_store,
            metadata=1,
        )


def test_fails_metadata_unknown_id(driver, function_store, existing_cube):
    df_extra = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]}
    )
    with pytest.raises(
        ValueError,
        match="Provided metadata for otherwise unspecified ktk_cube_dataset_ids: bar, foo",
    ):
        driver(
            data={"extra": df_extra},
            cube=existing_cube,
            store=function_store,
            metadata={"extra": {}, "foo": {}, "bar": {}},
        )


def test_fails_metadata_nested_wrong_type(driver, function_store, existing_cube):
    df_extra = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]}
    )
    with pytest.raises(
        TypeError,
        match="Provided metadata for dataset extra should be a dict but is int",
    ):
        driver(
            data={"extra": df_extra},
            cube=existing_cube,
            store=function_store,
            metadata={"extra": 1},
        )
