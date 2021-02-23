import pytest

from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.dask.common_cube import ensure_valid_cube_indices


class FakeSeedTableMetadata:
    names = ["d1", "d2", "p", "i1", "i2"]


class FakeExtraTableMetadata:
    names = ["d1", "p", "i1"]


def test_cube_with_valid_indices_is_not_modified_by_validation():
    """
    Test that a cube with valid indices is not modified by `ensure_valid_cube_indices`
    """
    source_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "source",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeSeedTableMetadata()},
            "partition_keys": ["p"],
            "indices": {
                "d1": {"1": ["part_1"]},
                "d2": {"1": ["part_1"]},
                "i1": {"1": ["part_1"]},
            },
        }
    )
    extra_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "extra",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeExtraTableMetadata()},
            "partition_keys": ["p"],
            "indices": {"i1": {"1": ["part_1"]}},
        }
    )
    cube = Cube(
        dimension_columns=["d1", "d2"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1"],
    )

    validated_cube = ensure_valid_cube_indices(
        {"source": source_metadata, "extra": extra_metadata}, cube
    )

    assert validated_cube == cube


def test_existing_indices_are_added_when_missing_in_cube():
    """
    Test that indices already existing in the dataset are added to the validated cube
    """
    source_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "source",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeSeedTableMetadata()},
            "partition_keys": ["p"],
            "indices": {
                "d1": {"1": ["part_1"]},
                "d2": {"1": ["part_1"]},
                "i1": {"1": ["part_1"]},
                "i2": {"1": ["part_1"]},
            },
        }
    )
    extra_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "extra",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeExtraTableMetadata()},
            "partition_keys": ["p"],
            "indices": {"i1": {"1": ["part_1"]}},
        }
    )
    cube = Cube(
        dimension_columns=["d1", "d2"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1"],
    )

    validated_cube = ensure_valid_cube_indices(
        {"source": source_metadata, "extra": extra_metadata}, cube
    )

    assert validated_cube.index_columns == {"i1", "i2"}


def test_raises_when_cube_defines_index_not_in_dataset():
    """
    Test that a `ValueError` is raised when the cube defines an index that is not part of a dataset
    """
    source_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "source",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeSeedTableMetadata()},
            "partition_keys": ["p"],
            "indices": {
                "d1": {"1": ["part_1"]},
                "d2": {"1": ["part_1"]},
                "i1": {"1": ["part_1"]},
            },
        }
    )
    extra_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "extra",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeExtraTableMetadata()},
            "partition_keys": ["p"],
            "indices": {"i1": {"1": ["part_1"]}},
        }
    )
    cube = Cube(
        dimension_columns=["d1", "d2"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i2"],
    )

    with pytest.raises(ValueError):
        ensure_valid_cube_indices(
            {"source": source_metadata, "extra": extra_metadata}, cube
        )


def test_no_indices_are_suppressed_when_they_already_exist():
    """
    Test that no indicies marked as suppressed in the cube are actually suppressed when
    they are already present in the dataset
    """
    source_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "source",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeSeedTableMetadata()},
            "partition_keys": ["p"],
            "indices": {
                "d1": {"1": ["part_1"]},
                "d2": {"1": ["part_1"]},
                "i1": {"1": ["part_1"]},
            },
        }
    )
    extra_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "extra",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeExtraTableMetadata()},
            "partition_keys": ["p"],
            "indices": {"i1": {"1": ["part_1"]}},
        }
    )
    cube = Cube(
        dimension_columns=["d1", "d2"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        suppress_index_on=["d1", "d2"],
    )

    validated_cube = ensure_valid_cube_indices(
        {"source": source_metadata, "extra": extra_metadata}, cube
    )

    assert validated_cube.suppress_index_on == frozenset()
