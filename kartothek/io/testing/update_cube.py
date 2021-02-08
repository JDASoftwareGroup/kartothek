from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from kartothek.core.cube.conditions import C
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.eager import read_table
from kartothek.io.eager_cube import build_cube, extend_cube, query_cube
from kartothek.io.testing.utils import assert_num_row_groups
from kartothek.serialization._parquet import ParquetSerializer


def _write_cube(function_store) -> Tuple[pd.DataFrame, Cube]:
    """
    Write a cube with dimension column "x" and partition column "p"

    returns the 'source' and 'enrich' dataframes and the cube specification.
    """
    df_source = pd.DataFrame(
        {
            "i1": [10, 11, 12, 13],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "x": [0, 1, 2, 3],
        }
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={"source": df_source},
        cube=cube,
        store=function_store,
        metadata={"source": {"meta_at_create": "data"}},
    )
    return df_source, cube


def _extend_cube(cube, function_store) -> pd.DataFrame:
    # extend the existing cube by a dataset 'ex' with columns a = x + 1000
    df = pd.DataFrame({"a": [1000, 1001], "p": [0, 1], "x": [0, 2]})
    extend_cube({"ex": df}, cube, function_store)
    return df


@pytest.mark.parametrize(
    "remove_partitions,new_partitions",
    [
        # only append:
        ([], [4, 5]),
        # do nothing:
        ([], []),
        # partial overwrite with new data for p=0
        ([0], [0, 1, 4]),
        # explicitly remove p=0 without overwriting it
        ([0], [1, 4]),
        # overwrite all:
        ([0, 1], [0, 1]),
    ],
)
def test_update_partitions(driver, function_store, remove_partitions, new_partitions):
    df_source, cube = _write_cube(function_store)

    df_source_new = pd.DataFrame(
        {
            "i1": range(200, 200 + len(new_partitions)),
            "p": np.array(new_partitions, np.int64),
            "v1": range(300, 300 + len(new_partitions)),
            "x": range(100, 100 + len(new_partitions)),
        }
    )

    # what should remain of the old data:
    df_source_of_old = df_source.loc[~df_source["p"].isin(set(remove_partitions))]
    df_source_expected_after = pd.concat(
        [df_source_of_old, df_source_new], sort=False, ignore_index=True
    )

    remove_conditions = C("p").isin(remove_partitions)

    result = driver(
        data={"source": df_source_new},
        remove_conditions=remove_conditions,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids={"source"},
        metadata={"source": {"some_new_meta": 42}},
    )

    assert set(result.keys()) == {"source"}

    dm_source_after = DatasetMetadata.load_from_store(
        cube.ktk_dataset_uuid("source"), function_store(), load_all_indices=True
    )

    assert "some_new_meta" in dm_source_after.metadata
    assert "meta_at_create" in dm_source_after.metadata

    # check values for "p" are as expected:
    expected_p_source = (set(df_source["p"].unique()) - set(remove_partitions)) | set(
        new_partitions
    )
    assert set(dm_source_after.indices["p"].index_dct) == expected_p_source

    df_read = query_cube(cube, function_store)[0]

    assert set(df_read.columns) == set(df_source_expected_after.columns)

    for df in (df_read, df_source_expected_after):
        df.sort_values("x", inplace=True)
        df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(df_read, df_source_expected_after)


@pytest.mark.parametrize("chunk_size_build", [None, 2])
@pytest.mark.parametrize("chunk_size_update", [None, 2])
def test_rowgroups_are_applied_when_df_serializer_is_passed_to_update_cube(
    driver, function_store, chunk_size_build, chunk_size_update
):
    """
    Test that the dataset is split into row groups depending on the chunk size

    Partitions build with ``chunk_size=None`` should keep a single row group if they
    are not touched by the update. Partitions that are newly created or replaced with
    ``chunk_size>0`` should be split into row groups accordingly.
    """
    # Build cube
    df = pd.DataFrame(data={"x": [0, 1], "p": [0, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df,
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(chunk_size=chunk_size_build),
    )

    # Update cube - replace p=1 and append p=2 partitions
    df_update = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [1, 1, 2, 2]}, columns=["x", "p"],
    )
    result = driver(
        data={"seed": df_update},
        remove_conditions=(C("p") == 1),  # Remove p=1 partition
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(chunk_size=chunk_size_update),
    )
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 1, 1: 2, 2: 2}
    part_chunk_size = {
        0: chunk_size_build,
        1: chunk_size_update,
        2: chunk_size_update,
    }

    assert len(dataset.partitions) == 3
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_single_rowgroup_when_df_serializer_is_not_passed_to_update_cube(
    driver, function_store
):
    """
    Test that the dataset has a single row group as default path
    """
    # Build cube
    df = pd.DataFrame(data={"x": [0, 1], "p": [0, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df, cube=cube, store=function_store,
    )

    # Update cube - replace p=1 and append p=2 partitions
    df_update = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [1, 1, 2, 2]}, columns=["x", "p"],
    )
    result = driver(
        data={"seed": df_update},
        remove_conditions=(C("p") == 1),  # Remove p=1 partition
        cube=cube,
        store=function_store,
    )
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 1, 1: 2, 2: 2}
    part_chunk_size = {0: None, 1: None, 2: None}

    assert len(dataset.partitions) == 3
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_compression_is_compatible_on_update_cube(driver, function_store):
    """
    Test that partitons written with different compression algorithms are compatible

    The compression algorithms are not parametrized because their availability depends
    on the arrow build. 'SNAPPY' and 'GZIP' are already assumed to be available in parts
    of the code. A fully parametrized test would also increase runtime and test complexity
    unnecessarily.
    """
    # Build cube
    df = pd.DataFrame(data={"x": [0, 1], "p": [0, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df,
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(compression="SNAPPY"),
    )

    # Update cube - replace p=1 and append p=2 partitions
    df_update = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [1, 1, 2, 2]}, columns=["x", "p"],
    )
    result = driver(
        data={"seed": df_update},
        remove_conditions=(C("p") == 1),  # Remove p=1 partition
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(compression="GZIP"),
    )
    dataset = result["seed"].load_all_indices(function_store())

    assert len(dataset.partitions) == 3


@pytest.mark.parametrize(
    "ktk_cube_dataset_ids", [{"source", "ex"}, {"source"}, {"ex"}, set()]
)
def test_update_respects_ktk_cube_dataset_ids(
    driver, function_store, ktk_cube_dataset_ids
):
    df_source, cube = _write_cube(function_store)
    df_ex = _extend_cube(cube, function_store)

    remove_conditions = C("p") == 0

    # This implicitly also tests that `data={}` behaves as expected and still deletes partitions
    # as requested via ktk_cube_dataset_ids and remove_conditions
    result = driver(
        data={},
        remove_conditions=remove_conditions,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
    )
    assert set(result) == ktk_cube_dataset_ids
    df_read = query_cube(cube, function_store)[0]

    # expected result: df_source left joined with df_ex; choosing the subset of p!=0 from each
    # that is in `ktk_cube_dataset_ids`:
    if "source" in ktk_cube_dataset_ids:
        df_source = df_source.loc[df_source["p"] != 0]
    if "ex" in ktk_cube_dataset_ids:
        df_ex = df_ex.loc[df_ex["p"] != 0]
    df_expected = df_source.merge(df_ex[["x", "a"]], how="left", on="x")
    df_expected = df_expected[sorted(df_expected.columns)]
    pd.testing.assert_frame_equal(df_read, df_expected)

    # test "ex" separately, because the test above based on the *left* merge does not tell us much about
    # "ex" in case the partitions were removed from "source"
    df_ex_read = read_table(cube.ktk_dataset_uuid("ex"), function_store)
    if "ex" in ktk_cube_dataset_ids:
        assert set(df_ex_read["p"]) == {1}
    else:
        assert set(df_ex_read["p"]) == {0, 1}


def test_cube_update_secondary_indices_subset(function_store, driver):

    cube1 = Cube(
        dimension_columns=["A"],
        partition_columns=["P"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["indexed"],
    )
    df_1 = pd.DataFrame({"A": range(10), "P": 1, "indexed": 1, "not-indexed": 1})
    build_cube(
        data={"source": df_1},
        cube=cube1,
        store=function_store,
        metadata={"source": {"meta_at_create": "data"}},
    )

    cube2 = Cube(
        dimension_columns=["A"],
        partition_columns=["P"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    df_2 = pd.DataFrame({"A": range(10, 20), "P": 1, "indexed": 2, "not-indexed": 1})
    driver(
        data={"source": df_2}, cube=cube2, store=function_store, remove_conditions=None
    )

    dataset_uuid = cube2.ktk_dataset_uuid(cube2.seed_dataset)
    dm = DatasetMetadata.load_from_store(
        dataset_uuid, function_store(), load_all_indices=True
    )
    obs_values = dm.indices["indexed"].observed_values()

    assert sorted(obs_values) == [1, 2]

    cube2 = Cube(
        dimension_columns=["A"],
        partition_columns=["P"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["not-indexed"],
    )
    with pytest.raises(
        ValueError,
        match='ExplicitSecondaryIndex or PartitionIndex "not-indexed" is missing in dataset',
    ):
        driver(
            data={"source": df_2},
            cube=cube2,
            store=function_store,
            remove_conditions=None,
        )


def test_cube_blacklist_dimension_index(function_store, driver):

    cube1 = Cube(
        dimension_columns=["A", "B"],
        partition_columns=["P"],
        uuid_prefix="cube",
        seed_dataset="source",
    )
    df_1 = pd.DataFrame({"A": range(10), "P": 1, "B": 1, "payload": ""})
    build_cube(
        data={"source": df_1},
        cube=cube1,
        store=function_store,
        metadata={"source": {"meta_at_create": "data"}},
    )

    cube2 = Cube(
        dimension_columns=["A", "B"],
        partition_columns=["P"],
        uuid_prefix="cube",
        seed_dataset="source",
        suppress_index_on=["B"],
    )
    df_2 = pd.DataFrame({"A": range(10), "P": 1, "B": 2, "payload": ""})
    driver(
        data={"source": df_2}, cube=cube2, store=function_store, remove_conditions=None
    )

    dataset_uuid = cube2.ktk_dataset_uuid(cube2.seed_dataset)
    dm = DatasetMetadata.load_from_store(
        dataset_uuid, function_store(), load_all_indices=True
    )
    obs_values = dm.indices["B"].observed_values()

    assert sorted(obs_values) == [1, 2]
