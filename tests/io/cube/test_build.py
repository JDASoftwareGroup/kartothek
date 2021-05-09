import uuid

import dask
import dask.bag as db
import dask.core
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest
from pandas.arrays import SparseArray
from tests.io.cube.utils import wrap_bag_write, wrap_ddf_write

from kartothek.core.cube.constants import (
    KTK_CUBE_DF_SERIALIZER,
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
    KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
from kartothek.io.dask.bag_cube import build_cube_from_bag
from kartothek.io.dask.dataframe_cube import build_cube_from_dataframe
from kartothek.io.eager_cube import build_cube
from kartothek.io_components.cube.write import MultiTableCommitAborted
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.serialization._parquet import ParquetSerializer

from .utils import assert_num_row_groups


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_write(build_cube_from_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_write(build_cube_from_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        return wrap_ddf_write(build_cube_from_dataframe)
    elif driver_name == "eager":
        return build_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))


def _count_execution_to_store(obj, store):
    store = store()
    key = "counter.{}".format(uuid.uuid4().hex)
    store.put(key, b"")
    return obj


def test_dask_bag_fusing(driver, function_store, driver_name, skip_eager):
    """
    There were two issues with the dask.bag write path.

    Ideal
    -----
    With 4 partitions and 2 datasets to write, it should look like this:

        o-+
          +
        o-+
          +-o
        o-+
          +
        o-+

    Missing linear fusing
    ---------------------
    The bags did not have linear fusing:

        o-o-o-o-o-+
                  +
        o-o-o-o-o-+
                  +-o
        o-o-o-o-o-+
                  +
        o-o-o-o-o-+

    Process-then-write instead of one-at-the-time
    ---------------------------------------------
    Due to the implementation of using 1 write bag per dataset and a pluck/split operation, the data for the whole bag
    partition was kept, then split, then written. Instead we aim for processing (including write) each DF in the
    partition and then move all metadata to the correct write path:

        o-s>-+
          v  |
          |  |
        o-------s>-+
          |  |  v  |
          |  |  |  |
        o-------------s>-+
          |  |  |  |  v  |
          |  |  |  |  |  |
        o-------------------s--+
          |  |  |  |  |  |  |  |
          +-----+-----+-----+-----o--+
             |     |     |     |     +-o
             +-----+-----+-----+--o--+
    """

    partition_size = 1 if driver_name == "dask_bag_bs1" else 3
    n_partitions = 4

    dfs = [
        {
            "source": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v1": 42}),
            "enrich": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v2": 1337}),
        }
        for i in range(partition_size * n_partitions)
    ]

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    bag = db.from_sequence(dfs, partition_size=partition_size).map(
        _count_execution_to_store, store=function_store
    )
    bag = build_cube_from_bag(
        data=bag,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids=["source", "enrich"],
    )
    dct = dask.optimize(bag)[0].__dask_graph__()
    tasks = {k for k, v in dct.items() if dask.core.istask(v)}
    assert len(tasks) == (n_partitions + 1)


def test_function_executed_once(driver, function_store, driver_name, skip_eager):
    """
    Test that the payload function is only executed once per branch.

    This was a bug in the dask_bag backend.
    """
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    df_source1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v1": [10, 11]})
    df_source2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v1": [12, 13]})
    df_enrich1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v2": [20, 21]})
    df_enrich2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v2": [22, 23]})

    dfs = [
        {"source": df_source1, "enrich": df_enrich1},
        {"source": df_source2, "enrich": df_enrich2},
    ]

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    if driver_name in ("dask_bag_bs1", "dask_bag_bs3"):
        bag = db.from_sequence(
            dfs, partition_size=1 if driver_name == "dask_bag_bs1" else 3
        ).map(_count_execution_to_store, store=function_store)
        bag = build_cube_from_bag(
            data=bag,
            cube=cube,
            store=function_store,
            ktk_cube_dataset_ids=["source", "enrich"],
        )
        bag.compute()
    else:
        raise ValueError("Missing implementation for driver: {}".format(driver_name))

    assert len(function_store().keys(prefix="counter.")) == 2


def test_simple_seed_only(driver, function_store):
    """
    Simple integration test w/ a seed dataset only. This is the most simple way to create a cube.
    """
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    result = driver(data=df, cube=cube, store=function_store)

    assert set(result.keys()) == {cube.seed_dataset}

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert ds.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert len(ds.partitions) == 2

    assert set(ds.indices.keys()) == {"p", "x"}
    assert isinstance(ds.indices["p"], PartitionIndex)
    assert isinstance(ds.indices["x"], ExplicitSecondaryIndex)

    assert ds.table_name == SINGLE_TABLE


def test_simple_two_datasets(driver, function_store):
    """
    Simple intergration test w/ 2 datasets.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich}, cube=cube, store=function_store
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 2

    assert set(ds_source.indices.keys()) == {"p", "x"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)

    assert set(ds_enrich.indices.keys()) == {"p"}
    assert isinstance(ds_enrich.indices["p"], PartitionIndex)

    assert ds_source.table_name == SINGLE_TABLE
    assert ds_enrich.table_name == SINGLE_TABLE


def test_indices(driver, function_store):
    """
    Test that index structures are created correctly.
    """
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [100, 101, 102, 103],
        }
    )
    df_enrich = pd.DataFrame(
        {
            "x": [0, 1, 4, 5],
            "p": [0, 0, 2, 2],
            "v2": [20, 21, 22, 23],
            "i2": [200, 201, 202, 203],
        }
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2"],
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich}, cube=cube, store=function_store
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert set(ds_source.indices.keys()) == {"p", "x", "i1"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)
    assert isinstance(ds_source.indices["i1"], ExplicitSecondaryIndex)

    assert set(ds_enrich.indices.keys()) == {"p", "i2"}
    assert isinstance(ds_enrich.indices["p"], PartitionIndex)
    assert isinstance(ds_enrich.indices["i2"], ExplicitSecondaryIndex)


def test_dimension_index_suppression(driver, function_store):
    """
    Test that suppress_index_on works as expected
    """
    df_source = pd.DataFrame(
        {
            "x": [0, 0, 1, 1],
            "y": [10, 11, 12, 13],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [100, 101, 102, 103],
        }
    )
    cube = Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2"],
        suppress_index_on=["x"],
    )
    result = driver(data={"source": df_source}, cube=cube, store=function_store)

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())

    assert set(ds_source.indices.keys()) == {"p", "i1", "y"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["i1"], ExplicitSecondaryIndex)
    assert isinstance(ds_source.indices["y"], ExplicitSecondaryIndex)


def test_do_not_modify_df(driver, function_store):
    """
    Functions should not modify their inputs.
    """
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    df_backup = df.copy()
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    driver(data=df, cube=cube, store=function_store)

    pdt.assert_frame_equal(df, df_backup)


@pytest.mark.filterwarnings("ignore::UnicodeWarning")
def test_parquet(driver, function_store):
    """
    Ensure the parquet files we generate are properly normalized.
    """
    df = pd.DataFrame(
        data={
            "x": [10, 1, 1, 0, 0],
            "y": [10, 0, 1, 1, 0],
            "p": [0, 1, 1, 1, 1],
            "föö".encode("utf8"): [100, 10, 11, 12, 13],
            "v": np.nan,
        },
        index=[0, 1000, 1001, 1002, 1003],
        columns=["x", "y", "p", "föö".encode("utf8"), "v"],
    )

    cube = Cube(
        dimension_columns=["x", "y"], partition_columns=["p"], uuid_prefix="cube"
    )
    result = driver(data=df, cube=cube, store=function_store)

    assert set(result.keys()) == {cube.seed_dataset}

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert len(ds.partitions) == 2
    for p in (0, 1):
        part_key = ds.indices["p"].index_dct[p][0]
        part = ds.partitions[part_key]
        key = part.files[SINGLE_TABLE]

        df_actual = KTK_CUBE_DF_SERIALIZER.restore_dataframe(function_store(), key)
        df_expected = (
            df.loc[df["p"] == p]
            .sort_values(["x", "y"])
            .reset_index(drop=True)
            .drop(columns=["p"])
            .rename(columns={"föö".encode("utf8"): "föö"})
        )

        pdt.assert_frame_equal(df_actual.reset_index(drop=True), df_expected)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_rowgroups_are_applied_when_df_serializer_is_passed_to_build_cube(
    driver, function_store, chunk_size
):
    """
    Test that the dataset is split into row groups depending on the chunk size
    """
    df = pd.DataFrame(data={"x": [0, 1, 2, 3], "p": [0, 1, 1, 1]}, columns=["x", "p"],)

    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    result = driver(
        data=df,
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(chunk_size=chunk_size),
    )
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 1, 1: 3}
    part_chunk_size = {0: chunk_size, 1: chunk_size}

    assert len(dataset.partitions) == 2
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_single_rowgroup_when_df_serializer_is_not_passed_to_build_cube(
    driver, function_store
):
    """
    Test that the dataset has a single row group as default path
    """
    df = pd.DataFrame(data={"x": [0, 1, 2, 3], "p": [0, 1, 1, 1]}, columns=["x", "p"],)

    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    result = driver(data=df, cube=cube, store=function_store,)
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 1, 1: 3}
    part_chunk_size = {0: None, 1: None}

    assert len(dataset.partitions) == 2
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_fail_sparse(driver, driver_name, function_store):
    """
    Ensure that sparse dataframes are rejected.
    """
    df = pd.DataFrame(
        data={
            "x": SparseArray([10, 1, 1, 0, 0]),
            "y": SparseArray([10, 0, 1, 1, 0]),
            "p": SparseArray([0, 1, 1, 1, 1]),
            "v": SparseArray([np.nan] * 5),
        }
    )

    cube = Cube(
        dimension_columns=["x", "y"], partition_columns=["p"], uuid_prefix="cube"
    )
    with pytest.raises(TypeError, match="Sparse data is not supported."):
        driver(data=df, cube=cube, store=function_store)


def test_metadata(driver, function_store):
    """
    Test auto- and user-generated metadata.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 4, 5], "p": [0, 0, 2, 2], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        metadata={"enrich": {"foo": 1}},
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset]
    assert set(ds_source.metadata.keys()) == {
        "creation_time",
        KTK_CUBE_METADATA_DIMENSION_COLUMNS,
        KTK_CUBE_METADATA_KEY_IS_SEED,
        KTK_CUBE_METADATA_PARTITION_COLUMNS,
        KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    }
    assert ds_source.metadata[KTK_CUBE_METADATA_DIMENSION_COLUMNS] == list(
        cube.dimension_columns
    )
    assert ds_source.metadata[KTK_CUBE_METADATA_KEY_IS_SEED] is True
    assert ds_source.metadata[KTK_CUBE_METADATA_PARTITION_COLUMNS] == list(
        cube.partition_columns
    )
    assert ds_source.metadata[KTK_CUBE_METADATA_SUPPRESS_INDEX_ON] == []

    ds_enrich = result["enrich"]
    assert set(ds_enrich.metadata.keys()) == {
        "creation_time",
        KTK_CUBE_METADATA_DIMENSION_COLUMNS,
        KTK_CUBE_METADATA_KEY_IS_SEED,
        KTK_CUBE_METADATA_PARTITION_COLUMNS,
        KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
        "foo",
    }
    assert ds_enrich.metadata[KTK_CUBE_METADATA_DIMENSION_COLUMNS] == list(
        cube.dimension_columns
    )
    assert ds_enrich.metadata[KTK_CUBE_METADATA_KEY_IS_SEED] is False
    assert ds_enrich.metadata[KTK_CUBE_METADATA_PARTITION_COLUMNS] == list(
        cube.partition_columns
    )
    assert ds_enrich.metadata["foo"] == 1
    assert ds_source.metadata[KTK_CUBE_METADATA_SUPPRESS_INDEX_ON] == []


def test_fails_metadata_wrong_type(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(
        TypeError, match="Provided metadata should be a dict but is int"
    ):
        driver(data={"source": df_source}, cube=cube, store=function_store, metadata=1)


def test_fails_metadata_unknown_id(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(
        ValueError,
        match="Provided metadata for otherwise unspecified ktk_cube_dataset_ids: bar, foo",
    ):
        driver(
            data={"source": df_source},
            cube=cube,
            store=function_store,
            metadata={"source": {}, "foo": {}, "bar": {}},
        )


def test_fails_metadata_nested_wrong_type(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(
        TypeError,
        match="Provided metadata for dataset source should be a dict but is int",
    ):
        driver(
            data={"source": df_source},
            cube=cube,
            store=function_store,
            metadata={"source": 1},
        )


def test_fails_missing_seed(driver, function_store):
    """
    A cube must contain its seed dataset, check this constraint as early as possible.
    """
    df = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v": [10, 11]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    with pytest.raises(ValueError) as exc:
        driver(data={"foo": df}, cube=cube, store=function_store)
    assert 'Seed data ("seed") is missing.' in str(exc.value)
    assert list(function_store().keys()) == []


def test_fails_missing_dimension_columns(driver, function_store):
    """
    Ensure that we catch missing dimension columns early.
    """
    df_source = pd.DataFrame({"x": [0, 1], "p": 0})
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(ValueError) as exc:
        driver(data=df_source, cube=cube, store=function_store)
    assert 'Missing dimension columns in seed data "source": y, z' in str(exc.value)
    assert list(function_store().keys()) == []


def test_fails_no_dimension_columns(driver, function_store):
    """
    Ensure that we catch missing dimension columns early.
    """
    df_source = pd.DataFrame({"x": [0, 1], "y": [0, 1], "z": [0, 1], "p": 0})
    df_enrich = pd.DataFrame({"p": [0], "v1": 0})
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(ValueError) as exc:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
        )
    assert (
        'Dataset "enrich" must have at least 1 of the following dimension columns: x, y'
        in str(exc.value)
    )
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_fails_duplicate_columns(driver, function_store, driver_name):
    """
    Catch weird pandas behavior.
    """
    if driver_name == "dask_dataframe":
        pytest.skip("already detected by dask.dataframe")

    df = pd.DataFrame(
        {"x": [0, 1], "p": 0, "a": 1, "b": 2}, columns=["x", "p", "a", "b"]
    ).rename(columns={"b": "a"})
    assert len(df.columns) == 4
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)
    assert 'Duplicate columns found in dataset "seed": x, p, a, a' in str(exc.value)
    assert list(function_store().keys()) == []


def test_fails_missing_partition_columns(driver, function_store):
    """
    Just make the Kartothek error nicer.
    """
    df = pd.DataFrame({"x": [0, 1], "p": 0})
    cube = Cube(
        dimension_columns=["x"], partition_columns=["p", "q", "r"], uuid_prefix="cube"
    )
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)
    assert 'Missing partition columns in dataset "seed": q, r' in str(exc.value)
    assert list(function_store().keys()) == []


def test_overwrite(driver, function_store):
    """
    Test overwrite behavior aka call the build function if the cube already exists.
    """
    df1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v": [10, 11]})
    df2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v": [12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    driver(data=df1, cube=cube, store=function_store)

    # implicit overwrite fails
    keys = set(function_store().keys())
    with pytest.raises(RuntimeError) as exc:
        driver(data=df1, cube=cube, store=function_store)
    assert "already exists and overwrite is not permitted" in str(exc.value)
    assert set(function_store().keys()) == keys

    # explicit overwrite works
    result = driver(data=df2, cube=cube, store=function_store, overwrite=True)

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert len(ds.partitions) == 1

    assert set(ds.indices["p"].index_dct.keys()) == {1}


def test_split(driver, function_store):
    """
    Imagine the user already splits the data.
    """
    df_source1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v1": [10, 11]})
    df_source2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v1": [12, 13]})
    df_enrich = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v2": [20, 21]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data=[{"source": df_source1, "enrich": df_enrich}, df_source2],
        cube=cube,
        store=function_store,
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 1


@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_df(driver, function_store, empty_first):
    """
    Might happen during DB queries.
    """
    df1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v1": [10, 11]})
    df2 = df1.loc[[]]
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data=[df2, df1] if empty_first else [df1, df2], cube=cube, store=function_store
    )
    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert len(ds.partitions) == 1
    assert (
        len(list(function_store().keys())) == 4
    )  # DS metadata, "x" index, common metadata, 1 partition


def test_fail_duplicates_local(driver, driver_name, function_store):
    """
    Might happen during DB queries.
    """
    if driver_name == "dask_dataframe":
        pytest.xfail(reason="Cannot guarantee duplicates for DDF")
    df = pd.DataFrame(
        {
            "x": [0, 0],
            "y": ["a", "a"],
            "z": [pd.Timestamp("2017"), pd.Timestamp("2017")],
            "p": [0, 0],
        }
    )
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)
    msg = """
Found duplicate cells by [p, x, y, z] in dataset "source", example:

Keys:
p                      0
x                      0
y                      a
z    2017-01-01 00:00:00

Identical Payload:
n/a

Non-Idential Payload:
n/a
""".strip()
    assert msg in str(exc.value)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_accept_projected_duplicates(driver, function_store):
    """
    Otherwise partitioning does not work w/ projected data.
    """
    df_seed = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
    df_enrich = pd.DataFrame({"x": [0, 1, 0, 1], "p": [0, 0, 1, 1]})
    cube = Cube(
        dimension_columns=["x", "y"], partition_columns=["p"], uuid_prefix="cube"
    )
    driver(data={"seed": df_seed, "enrich": df_enrich}, cube=cube, store=function_store)


@pytest.mark.xfail(
    strict=True, reason="Cannot be checked with current index structures."
)
def test_fail_duplicates_global(driver_name, driver, function_store):
    """
    Might happen due to bugs.
    """
    if driver_name == "eager":
        pytest.skip(reason="Problem does not occur in eager mode.")

    df1 = pd.DataFrame({"x": [0], "y": ["a"], "z": [pd.Timestamp("2017")], "p": [0]})
    df2 = pd.DataFrame({"x": [0], "y": ["a"], "z": [pd.Timestamp("2017")], "p": [1]})
    cube = Cube(
        dimension_columns=["x", "y", "z"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(ValueError):
        driver(data=[df1, df2], cube=cube, store=function_store)


def test_regression_pseudo_duplicates(driver, function_store):
    """
    Might happen due to bugs.
    """
    df = pd.DataFrame({"x": [0, 0, 2, 3], "y": [0, 1, 2, 2], "p": [0, 1, 0, 1]})
    cube = Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    driver(data=df, cube=cube, store=function_store)


def test_fail_wrong_types(driver, function_store):
    """
    Might catch nasty pandas and other type bugs.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert 'Found incompatible entries for column "x"' in str(cause)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_distinct_branches(driver, function_store):
    """
    Just check this actually works.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data=[{"source": df_source}, {"enrich": df_enrich}],
        cube=cube,
        store=function_store,
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 2


def test_fail_nondistinc_payload(driver, function_store):
    """
    This would lead to problems during the query phase.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
        )
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert "Found columns present in multiple datasets" in str(cause)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_fail_partial_overwrite(driver, function_store):
    """
    Either overwrite all or no datasets.
    """
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    df_source1 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich1 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    driver(
        data={"source": df_source1, "enrich": df_enrich1},
        cube=cube,
        store=function_store,
    )

    keys = set(function_store().keys())
    df_source2 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]}
    )
    with pytest.raises(ValueError) as exc:
        driver(
            data={"source": df_source2}, cube=cube, store=function_store, overwrite=True
        )
    assert (
        str(exc.value)
        == "Following datasets exists but are not overwritten (partial overwrite), this is not allowed: enrich"
    )
    assert set(function_store().keys()) == keys


def test_fail_partial_build(driver, function_store):
    """
    Either overwrite all or no datasets.
    """
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    df_source1 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich1 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    driver(
        data={"source": df_source1, "enrich": df_enrich1},
        cube=cube,
        store=function_store,
    )

    # delete everything that belongs to the seed dataset
    to_delete = {
        k
        for k in function_store().keys()
        if k.startswith(cube.ktk_dataset_uuid(cube.seed_dataset))
    }
    for k in to_delete:
        function_store().delete(k)

    keys = set(function_store().keys())
    df_source2 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v3": [10, 11, 12, 13]}
    )
    with pytest.raises(ValueError) as exc:
        driver(data={"source": df_source2}, cube=cube, store=function_store)
    assert (
        str(exc.value)
        == "Following datasets exists but are not overwritten (partial overwrite), this is not allowed: enrich"
    )
    assert set(function_store().keys()) == keys


def test_fails_projected_duplicates(driver, driver_name, function_store):
    """
    Test if duplicate check also works w/ projected data. (was a regression)
    """
    if driver_name == "dask_dataframe":
        pytest.xfail(reason="Cannot guarantee duplicates for DDF")
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
        }
    )
    df_enrich = pd.DataFrame(
        {"y": [0, 0, 1, 1], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23], "v3": 42}
    )
    cube = Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(ValueError) as exc:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
        )
    msg = """
Found duplicate cells by [p, y] in dataset "enrich", example:

Keys:
p    0
y    0

Identical Payload:
v3    42

Non-Idential Payload:
   v2
0  20
1  21
""".strip()
    assert msg in str(exc.value)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_projected_data(driver, function_store):
    """
    Projected dataset (useful for de-duplication).
    """
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
        }
    )
    df_enrich = pd.DataFrame({"y": [0, 1], "p": [0, 1], "v2": [20, 21]})
    cube = Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich}, cube=cube, store=function_store
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 2


def test_fails_null_dimension(driver, function_store):
    """
    Since we do not allow NULL values in queries, it should be banned from dimension columns in the first place.
    """
    df = pd.DataFrame(
        {"x": [0, 1, 2, np.nan], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)

    assert 'Found NULL-values in dimension column "x" of dataset "seed"' in str(exc)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("seed"), function_store())


def test_fails_null_partition(driver, function_store):
    """
    Since we do not allow NULL values in queries, it should be banned from partition columns in the first place.
    """
    df = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, np.nan], "v": [10, 11, 12, 13]}
    )
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)
    assert 'Found NULL-values in partition column "p" of dataset "seed"' in str(
        exc.value
    )
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("seed"), function_store())


def test_fails_null_index(driver, function_store):
    """
    Since we do not allow NULL values in queries, it should be banned from index columns in the first place.
    """
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v": [10, 11, 12, 13],
            "i1": [0, 1, 2, np.nan],
        }
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        index_columns=["i1"],
    )
    with pytest.raises(ValueError) as exc:
        driver(data=df, cube=cube, store=function_store)
    assert 'Found NULL-values in index column "i1"' in str(exc.value)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("seed"), function_store())


@pytest.mark.xfail(reason="different")
def test_fail_all_empty(driver, driver_name, function_store):
    """
    Might happen due to DB-based filters.
    """
    df = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]}
    ).loc[[]]
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(data=df, cube=cube, store=function_store)
    exc = exc_info.value.__cause__
    assert isinstance(exc, ValueError)
    assert "Cannot write empty datasets" in str(exc)

    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_overwrite_rollback_ktk_cube(driver, function_store):
    """
    Checks that require a rollback (like overlapping columns) should recover the former state correctly.
    """
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3", "i4"],
    )

    df_source1 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [10, 11, 12, 13],
        }
    )
    df_enrich1 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "i2": [20, 21, 22, 23],
            "v2": [20, 21, 22, 23],
        }
    )
    driver(
        data={"source": df_source1, "enrich": df_enrich1},
        cube=cube,
        store=function_store,
    )

    df_source2 = pd.DataFrame(
        {
            "x": [10, 11],
            "p": [10, 10],
            "v1": [10.0, 11.0],  # also use another dtype here (was int)
            "i3": [10, 11],
        }
    )
    df_enrich2 = pd.DataFrame(
        {"x": [10, 11], "p": [10, 10], "v1": [20, 21], "i4": [20, 21]}
    )
    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(
            data={"source": df_source2, "enrich": df_enrich2},
            cube=cube,
            store=function_store,
            overwrite=True,
        )
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause).startswith("Found columns present in multiple datasets:")

    ds_source = DatasetMetadata.load_from_store(
        uuid=cube.ktk_dataset_uuid("source"), store=function_store()
    ).load_all_indices(function_store())
    ds_enrich = DatasetMetadata.load_from_store(
        uuid=cube.ktk_dataset_uuid("enrich"), store=function_store()
    ).load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 2

    assert set(ds_source.indices.keys()) == {"p", "x", "i1"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)
    assert set(ds_source.indices["x"].index_dct.keys()) == {0, 1, 2, 3}
    assert set(ds_source.indices["i1"].index_dct.keys()) == {10, 11, 12, 13}

    assert set(ds_enrich.indices.keys()) == {"p", "i2"}
    assert isinstance(ds_enrich.indices["p"], PartitionIndex)
    assert set(ds_enrich.indices["i2"].index_dct.keys()) == {20, 21, 22, 23}

    assert ds_source.schema.field("v1").type == pa.int64()


@pytest.mark.parametrize("none_first", [False, True])
def test_nones(driver, function_store, none_first, driver_name):
    """
    Test what happens if user passes None to ktk_cube.
    """
    if driver_name == "dask_dataframe":
        pytest.skip("user cannot create None-partitions with dask.dataframe")

    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    result = driver(
        data=[None, df] if none_first else [df, None], cube=cube, store=function_store
    )

    assert set(result.keys()) == {cube.seed_dataset}

    ds = list(result.values())[0]
    ds = ds.load_all_indices(function_store())

    assert ds.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert len(ds.partitions) == 2

    assert set(ds.indices.keys()) == {"p", "x"}
    assert isinstance(ds.indices["p"], PartitionIndex)
    assert isinstance(ds.indices["x"], ExplicitSecondaryIndex)

    assert ds.table_name == SINGLE_TABLE


def test_fail_not_a_df(driver, function_store):
    """
    Pass some weird objects in.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(TypeError) as exc:
        driver(
            data={"source": df_source, "enrich": pd.Series(range(10))},
            cube=cube,
            store=function_store,
        )
    assert (
        'Provided DataFrame is not a pandas.DataFrame or None, but is a "Series"'
        in str(exc.value)
    )


def test_fail_wrong_dataset_ids(driver, function_store, skip_eager, driver_name):
    if driver_name == "dask_dataframe":
        pytest.skip("not an interface for dask.dataframe")

    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    with pytest.raises(ValueError) as exc:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
            ktk_cube_dataset_ids=["source", "extra"],
        )

    assert (
        'Ktk_cube Dataset ID "enrich" is present during pipeline execution but was not '
        "specified in ktk_cube_dataset_ids (extra, source)." in str(exc.value)
    )


def test_fail_no_store_factory(driver, function_store, skip_eager):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    store = function_store()
    with pytest.raises(TypeError) as exc:
        driver(data=df, cube=cube, store=store, no_run=True)
    assert str(exc.value) == "store must be a factory but is HFilesystemStore"


def test_fail_partition_on_1(driver, function_store):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v": [10, 11, 12, 13]})
    cube = Cube(
        dimension_columns=["x"], partition_columns=["p", "q"], uuid_prefix="cube"
    )

    with pytest.raises(ValueError) as exc_info:
        driver(
            data=df,
            cube=cube,
            store=function_store,
            partition_on={cube.seed_dataset: ["x", "p"]},
        )

    cause = exc_info.value  # .__cause__
    assert isinstance(cause, ValueError)
    assert (
        "Seed dataset seed must have the following, fixed partition-on attribute: p, q"
        in str(cause)
    )

    assert set(function_store().keys()) == set()


def test_fail_partition_on_3(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame({"x": [0, 1, 2, 3], "v2": [20, 21, 22, 23]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    with pytest.raises(
        ValueError,
        match="partition-on attribute of dataset enrich contains duplicates: p, p",
    ):
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
            partition_on={"enrich": ["p", "p"]},
        )

    assert set(function_store().keys()) == set()


def test_fail_partition_on_4(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    with pytest.raises(
        ValueError, match="Unspecified but provided partition columns in enrich: p"
    ):
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
            partition_on={"enrich": []},
        )
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())


def test_partition_on_enrich_none(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame({"x": [0, 1, 2, 3], "v2": [20, 21, 22, 23]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        partition_on={"enrich": []},
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 1

    assert set(ds_source.indices.keys()) == {"p", "x"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)

    assert set(ds_enrich.indices.keys()) == set()

    assert ds_source.table_name == SINGLE_TABLE
    assert ds_enrich.table_name == SINGLE_TABLE


def test_partition_on_enrich_extra(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        partition_on={"enrich": ["p", "x"]},
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 4

    assert set(ds_source.indices.keys()) == {"p", "x"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)

    assert set(ds_enrich.indices.keys()) == {"p", "x"}
    assert isinstance(ds_enrich.indices["p"], PartitionIndex)
    assert isinstance(ds_enrich.indices["x"], PartitionIndex)

    assert ds_source.table_name == SINGLE_TABLE
    assert ds_enrich.table_name == SINGLE_TABLE


def test_partition_on_index_column(driver, function_store):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame(
        {"x": [0, 1, 2, 3], "i": [0, 0, 1, 2], "v2": [20, 21, 22, 23]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        index_columns=["i"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    result = driver(
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        partition_on={"enrich": ["i"]},
    )

    assert set(result.keys()) == {cube.seed_dataset, "enrich"}

    ds_source = result[cube.seed_dataset].load_all_indices(function_store())
    ds_enrich = result["enrich"].load_all_indices(function_store())

    assert ds_source.uuid == cube.ktk_dataset_uuid(cube.seed_dataset)
    assert ds_enrich.uuid == cube.ktk_dataset_uuid("enrich")

    assert len(ds_source.partitions) == 2
    assert len(ds_enrich.partitions) == 3

    assert set(ds_source.indices.keys()) == {"p", "x"}
    assert isinstance(ds_source.indices["p"], PartitionIndex)
    assert isinstance(ds_source.indices["x"], ExplicitSecondaryIndex)

    assert set(ds_enrich.indices.keys()) == {"i"}
    assert isinstance(ds_enrich.indices["i"], PartitionIndex)

    assert ds_source.table_name == SINGLE_TABLE
    assert ds_enrich.table_name == SINGLE_TABLE


def test_fail_partition_on_nondistinc_payload(driver, function_store):
    """
    This would lead to problems during the query phase.
    """
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "v1": [10, 11, 12, 13]}
    )
    df_enrich = pd.DataFrame({"x": [0, 1, 2, 3], "v1": [20, 21, 22, 23]})
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    with pytest.raises(MultiTableCommitAborted) as exc_info:
        driver(
            data={"source": df_source, "enrich": df_enrich},
            cube=cube,
            store=function_store,
            partition_on={"enrich": ["v1"]},
        )
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert "Found columns present in multiple datasets" in str(cause)
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("source"), function_store())
    assert not DatasetMetadata.exists(cube.ktk_dataset_uuid("enrich"), function_store())
