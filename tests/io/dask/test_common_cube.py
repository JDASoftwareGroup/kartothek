from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.dask.common_cube import ensure_valid_cube_indices


class FakeSeedTableMetadata:
    names = ["d1", "d2", "p", "i"]


class FakeExtraTableMetadata:
    names = ["d1", "p", "i"]


def test_cube_indices_are_validated():
    source_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "source",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeSeedTableMetadata()},
            "partition_keys": ["p"],
            "indices": {
                "d1": {"1": ["part_1"]},
                "d2": {"1": ["part_1"]},
                "i": {"1": ["part_1"]},
            },
        }
    )
    extra_metadata = DatasetMetadata.from_dict(
        {
            "dataset_uuid": "extra",
            "dataset_metadata_version": 4,
            "table_meta": {"table": FakeExtraTableMetadata()},
            "partition_keys": ["p"],
            "indices": {"i": {"1": ["part_1"]}},
        }
    )
    cube = Cube(
        dimension_columns=["d1", "d2"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i"],
    )

    validated_cube = ensure_valid_cube_indices(
        {"source": source_metadata, "extra": extra_metadata}, cube
    )

    assert validated_cube == cube
