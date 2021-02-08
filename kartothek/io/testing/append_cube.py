import pandas as pd
import pytest

from kartothek.core.cube.constants import (
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
    KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.eager_cube import build_cube
from kartothek.io.testing.utils import assert_num_row_groups
from kartothek.serialization._parquet import ParquetSerializer

__all__ = (
    "existing_cube",
    "test_append_partitions",
    "test_append_partitions_no_ts",
    "test_compression_is_compatible_on_append_cube",
    "test_fails_incompatible_dtypes",
    "test_fails_missing_column",
    "test_fails_unknown_dataset",
    "test_indices",
    "test_metadata",
    "test_rowgroups_are_applied_when_df_serializer_is_passed_to_append_cube",
    "test_single_rowgroup_when_df_serializer_is_not_passed_to_append_cube",
)


@pytest.fixture
def existing_cube(function_store):
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [10, 11, 12, 13],
        }
    )
    df_enrich = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v2": [10, 11, 12, 13],
            "i2": [10, 11, 12, 13],
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
        data={"source": df_source, "enrich": df_enrich},
        cube=cube,
        store=function_store,
        metadata={"source": {"a": 10, "b": 11}, "enrich": {"a": 20, "b": 21}},
    )
    return cube


def test_append_partitions(driver, function_store, existing_cube):
    partitions_source_1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("source"), function_store()
        ).partitions.keys()
    )
    partitions_enrich_1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("enrich"), function_store()
        ).partitions.keys()
    )

    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [20, 21, 22, 23],
            "i1": [20, 21, 22, 23],
        }
    )

    result = driver(
        data={"source": df_source}, cube=existing_cube, store=function_store
    )

    assert set(result.keys()) == {"source"}

    ds_source = result["source"]
    ds_enrich = DatasetMetadata.load_from_store(
        existing_cube.ktk_dataset_uuid("enrich"), function_store()
    )

    partitions_source_2 = set(ds_source.partitions.keys())
    partitions_enrich_2 = set(ds_enrich.partitions.keys())

    assert len(partitions_source_2) > len(partitions_source_1)
    assert partitions_source_1.issubset(partitions_source_2)

    assert partitions_enrich_2 == partitions_enrich_1


@pytest.mark.parametrize("chunk_size_build", [None, 2])
@pytest.mark.parametrize("chunk_size_append", [None, 2])
def test_rowgroups_are_applied_when_df_serializer_is_passed_to_append_cube(
    driver, function_store, chunk_size_build, chunk_size_append
):
    """
    Test that the dataset is split into row groups depending on the chunk size

    Partitions build with ``chunk_size=None`` should keep a single row group after the append. Partitions that are newly created with
    ``chunk_size>0`` should be split into row groups accordingly.
    """

    # Build cube
    df = pd.DataFrame(data={"x": [0, 1, 2, 3], "p": [0, 0, 1, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df,
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(chunk_size=chunk_size_build),
    )

    # Append to cube
    df_append = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [2, 3, 3, 3]}, columns=["x", "p"],
    )
    result = driver(
        data={"seed": df_append},
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(chunk_size=chunk_size_append),
    )
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 2, 1: 2, 2: 1, 3: 3}
    part_chunk_size = {
        0: chunk_size_build,
        1: chunk_size_build,
        2: chunk_size_append,
        3: chunk_size_append,
    }

    assert len(dataset.partitions) == 4
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_single_rowgroup_when_df_serializer_is_not_passed_to_append_cube(
    driver, function_store
):
    """
    Test that the dataset has a single row group as default path
    """

    # Build cube
    df = pd.DataFrame(data={"x": [0, 1, 2, 3], "p": [0, 0, 1, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df, cube=cube, store=function_store,
    )

    # Append to cube
    df_append = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [2, 3, 3, 3]}, columns=["x", "p"],
    )
    result = driver(data={"seed": df_append}, cube=cube, store=function_store,)
    dataset = result["seed"].load_all_indices(function_store())

    part_num_rows = {0: 2, 1: 2, 2: 1, 3: 3}
    part_chunk_size = {0: None, 1: None, 2: None, 3: None}

    assert len(dataset.partitions) == 4
    assert_num_row_groups(function_store(), dataset, part_num_rows, part_chunk_size)


def test_compression_is_compatible_on_append_cube(driver, function_store):
    """
    Test that partitons written with different compression algorithms are compatible

    The compression algorithms are not parametrized because their availability depends
    on the arrow build. 'SNAPPY' and 'GZIP' are already assumed to be available in parts
    of the code. A fully parametrized test would also increase runtime and test complexity
    unnecessarily.
    """
    # Build cube
    df = pd.DataFrame(data={"x": [0, 1, 2, 3], "p": [0, 0, 1, 1]}, columns=["x", "p"],)
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="rg-cube")
    build_cube(
        data=df,
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(compression="SNAPPY"),
    )

    # Append to cube
    df_append = pd.DataFrame(
        data={"x": [0, 1, 2, 3], "p": [2, 3, 3, 3]}, columns=["x", "p"],
    )
    result = driver(
        data={"seed": df_append},
        cube=cube,
        store=function_store,
        df_serializer=ParquetSerializer(compression="GZIP"),
    )
    dataset = result["seed"].load_all_indices(function_store())

    assert len(dataset.partitions) == 4


def test_append_partitions_no_ts(driver, function_store):
    df_source1 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [10, 11, 12, 13],
        }
    )
    df_enrich1 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "v2": [10, 11, 12, 13], "i2": [10, 11, 12, 13]}
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={"source": df_source1, "enrich": df_enrich1},
        cube=cube,
        store=function_store,
        metadata={"source": {"a": 10, "b": 11}, "enrich": {"a": 20, "b": 21}},
        partition_on={"enrich": []},
    )

    partitions_source_1 = set(
        DatasetMetadata.load_from_store(
            cube.ktk_dataset_uuid("source"), function_store()
        ).partitions.keys()
    )
    partitions_enrich_1 = set(
        DatasetMetadata.load_from_store(
            cube.ktk_dataset_uuid("enrich"), function_store()
        ).partitions.keys()
    )

    df_source2 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [20, 21, 22, 23],
            "i1": [20, 21, 22, 23],
        }
    )
    df_enrich2 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "v2": [20, 21, 22, 23], "i2": [20, 21, 22, 23]}
    )

    result = driver(
        data={"source": df_source2, "enrich": df_enrich2},
        cube=cube,
        store=function_store,
    )

    assert set(result.keys()) == {"source", "enrich"}

    ds_source = result["source"]
    ds_enrich = result["enrich"]

    partitions_source_2 = set(ds_source.partitions.keys())
    partitions_enrich_2 = set(ds_enrich.partitions.keys())

    assert len(partitions_source_2) > len(partitions_source_1)
    assert partitions_source_1.issubset(partitions_source_2)

    assert len(partitions_enrich_2) > len(partitions_enrich_1)
    assert partitions_enrich_1.issubset(partitions_enrich_2)


def test_indices(driver, function_store, existing_cube):
    idx1_1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("source"), function_store()
        )
        .load_all_indices(function_store())
        .indices["i1"]
        .index_dct.keys()
    )
    idx2_1 = set(
        DatasetMetadata.load_from_store(
            existing_cube.ktk_dataset_uuid("enrich"), function_store()
        )
        .load_all_indices(function_store())
        .indices["i2"]
        .index_dct.keys()
    )

    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [20, 21, 22, 23],
            "i1": [20, 21, 22, 23],
        }
    )

    result = driver(
        data={"source": df_source}, cube=existing_cube, store=function_store
    )

    assert set(result.keys()) == {"source"}

    ds_source = result["source"]
    ds_enrich = DatasetMetadata.load_from_store(
        existing_cube.ktk_dataset_uuid("enrich"), function_store()
    )

    idx1_2 = set(
        ds_source.load_all_indices(function_store()).indices["i1"].index_dct.keys()
    )
    idx2_2 = set(
        ds_enrich.load_all_indices(function_store()).indices["i2"].index_dct.keys()
    )

    assert idx1_1.issubset(idx1_2)
    assert len(idx1_1) < len(idx1_2)

    assert idx2_1 == idx2_2


def test_fails_incompatible_dtypes(driver, function_store, existing_cube):
    """
    Should also cross check w/ seed dataset.
    """
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10.0, 11.0, 12.0, 13.0],
            "i1": [10, 11, 12, 13],
        }
    )

    with pytest.raises(ValueError, match="Schema violation"):
        driver(data={"source": df_source}, cube=existing_cube, store=function_store)


def test_fails_missing_column(driver, function_store, existing_cube):
    df_source = pd.DataFrame(
        {"x": [0, 1, 2, 3], "p": [0, 0, 1, 1], "i1": [10, 11, 12, 13]}
    )

    with pytest.raises(ValueError, match="Schema violation"):
        driver(data={"source": df_source}, cube=existing_cube, store=function_store)


def test_fails_unknown_dataset(driver, function_store, existing_cube):
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "i1": [10, 11, 12, 13],
        }
    )
    df_zoo = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v3": [10, 11, 12, 13],
            "i3": [10, 11, 12, 13],
        }
    )

    keys_pre = set(function_store().keys())

    with pytest.raises(ValueError, match="Unknown / non-existing datasets: zoo"):
        driver(
            data={"source": df_source, "zoo": df_zoo},
            cube=existing_cube,
            store=function_store,
        )

    keys_post = set(function_store().keys())
    assert keys_pre == keys_post


def test_metadata(driver, function_store, existing_cube):
    """
    Test auto- and user-generated metadata.
    """
    df_source = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "p": [0, 0, 1, 1],
            "v1": [20, 21, 22, 23],
            "i1": [20, 21, 22, 23],
        }
    )

    result = driver(
        data={"source": df_source},
        cube=existing_cube,
        store=function_store,
        metadata={"source": {"a": 12, "c": 13}},
    )

    assert set(result.keys()) == {"source"}

    ds_source = result["source"]
    assert set(ds_source.metadata.keys()) == {
        "a",
        "b",
        "c",
        "creation_time",
        KTK_CUBE_METADATA_DIMENSION_COLUMNS,
        KTK_CUBE_METADATA_KEY_IS_SEED,
        KTK_CUBE_METADATA_PARTITION_COLUMNS,
        KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    }
    assert ds_source.metadata["a"] == 12
    assert ds_source.metadata["b"] == 11
    assert ds_source.metadata["c"] == 13
    assert ds_source.metadata[KTK_CUBE_METADATA_DIMENSION_COLUMNS] == list(
        existing_cube.dimension_columns
    )
    assert ds_source.metadata[KTK_CUBE_METADATA_KEY_IS_SEED] is True
    assert ds_source.metadata[KTK_CUBE_METADATA_PARTITION_COLUMNS] == list(
        existing_cube.partition_columns
    )
    assert ds_source.metadata[KTK_CUBE_METADATA_SUPPRESS_INDEX_ON] == []

    ds_enrich = DatasetMetadata.load_from_store(
        existing_cube.ktk_dataset_uuid("enrich"), function_store()
    )
    assert set(ds_enrich.metadata.keys()) == {
        "a",
        "b",
        "creation_time",
        KTK_CUBE_METADATA_DIMENSION_COLUMNS,
        KTK_CUBE_METADATA_KEY_IS_SEED,
        KTK_CUBE_METADATA_PARTITION_COLUMNS,
        KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    }
    assert ds_enrich.metadata["a"] == 20
    assert ds_enrich.metadata["b"] == 21
    assert ds_enrich.metadata[KTK_CUBE_METADATA_DIMENSION_COLUMNS] == list(
        existing_cube.dimension_columns
    )
    assert ds_enrich.metadata[KTK_CUBE_METADATA_KEY_IS_SEED] is False
    assert ds_enrich.metadata[KTK_CUBE_METADATA_PARTITION_COLUMNS] == list(
        existing_cube.partition_columns
    )
    assert ds_source.metadata[KTK_CUBE_METADATA_SUPPRESS_INDEX_ON] == []
