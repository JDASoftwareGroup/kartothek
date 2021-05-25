# -*- coding: utf-8 -*-


import pytest
import simplejson

import kartothek.core._zmsgpack as msgpack
from kartothek.core.dataset import DatasetMetadata, DatasetMetadataBuilder
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.partition import Partition
from kartothek.core.testing import TIME_TO_FREEZE_ISO


@pytest.mark.parametrize("explicit_partitions", [True, False])
def test_builder_empty(explicit_partitions, metadata_version, frozen_time):
    creation_time = TIME_TO_FREEZE_ISO
    expected = {
        "dataset_uuid": "uuid",
        "dataset_metadata_version": metadata_version,
        "metadata": {"creation_time": creation_time},
    }
    if explicit_partitions:
        expected["partitions"] = {}
    key, result = DatasetMetadataBuilder(
        "uuid",
        metadata_version=metadata_version,
        explicit_partitions=explicit_partitions,
    ).to_json()
    result = simplejson.loads(result)
    assert key == "uuid.by-dataset-metadata.json"
    assert result == expected


def test_builder_msgpack(metadata_version, frozen_time):
    creation_time = TIME_TO_FREEZE_ISO
    expected = {
        "dataset_uuid": "uuid",
        "dataset_metadata_version": metadata_version,
        "metadata": {"creation_time": creation_time},
        "partitions": {},
    }
    key, result = DatasetMetadataBuilder(
        "uuid", metadata_version=metadata_version
    ).to_msgpack()
    result = msgpack.unpackb(result)
    assert key == "uuid.by-dataset-metadata.msgpack.zstd"
    assert result == expected


def test_builder_to_dataset(metadata_version, frozen_time):
    expected = {
        "dataset_uuid": "uuid",
        "dataset_metadata_version": metadata_version,
        "partitions": {"part_2": {"files": {"core": "uuid/core/part_2.parquet"}}},
        "metadata": {"key": "value", "creation_time": TIME_TO_FREEZE_ISO},
        "indices": {"col1": {"a": ["part1"], "b": ["part2"]}},
    }

    builder = DatasetMetadataBuilder("uuid", metadata_version=metadata_version)
    part_2 = Partition("part_2", {"core": "uuid/core/part_2.parquet"})
    builder.add_partition("part_2", part_2)
    builder.add_metadata("key", "value")
    builder.add_embedded_index(
        "col1", ExplicitSecondaryIndex("col1", {"a": ["part1"], "b": ["part2"]})
    )

    result = builder.to_dataset()
    expected_from_dict = DatasetMetadata.from_dict(expected)
    assert result == expected_from_dict


def test_builder_full(metadata_version, frozen_time):
    expected = {
        "dataset_uuid": "uuid",
        "dataset_metadata_version": metadata_version,
        "partitions": {
            "run_id=1/L=1/P=1/part_1": {
                "files": {
                    "core": "uuid/core/run_id=1/L=1/P=1/part_1.parquet",
                    "helper": "uuid/helper/run_id=1/L=1/P=1/part_1.parquet",
                }
            }
        },
        "metadata": {"key": "value", "creation_time": TIME_TO_FREEZE_ISO},
        "indices": {
            "col1": {
                "a": ["run_id=1/L=1/P=1/part_1"],
                "b": ["run_id=2/L=1/P=1/part_1"],
            },
            "col2": "uuid.col2.by-dataset-index.parquet",
        },
        "partition_keys": ["L", "P"],
    }

    builder = DatasetMetadataBuilder(
        "uuid", metadata_version=metadata_version, partition_keys=["L", "P"]
    )
    part_2 = Partition(
        label="run_id=1/L=1/P=1/part_1",
        files={
            "core": "uuid/core/run_id=1/L=1/P=1/part_1.parquet",
            "helper": "uuid/helper/run_id=1/L=1/P=1/part_1.parquet",
        },
    )
    builder.add_partition("run_id=1/L=1/P=1/part_1", part_2)
    builder.add_metadata("key", "value")
    builder.add_external_index("col2")
    builder.add_embedded_index(
        "col1",
        ExplicitSecondaryIndex(
            "col1", {"a": ["run_id=1/L=1/P=1/part_1"], "b": ["run_id=2/L=1/P=1/part_1"]}
        ),
    )
    key, result = builder.to_json()
    result = simplejson.loads(result)
    assert key == "uuid.by-dataset-metadata.json"
    assert result == expected


def test_builder_empty_partition_keys(store, metadata_version, frozen_time):
    expected = {
        "dataset_uuid": "uuid",
        "dataset_metadata_version": metadata_version,
        "metadata": {"creation_time": TIME_TO_FREEZE_ISO},
        "partition_keys": ["L", "P"],
        "partitions": {},
    }

    builder = DatasetMetadataBuilder(
        "uuid", metadata_version=4, partition_keys=["L", "P"]
    )
    key, result = builder.to_json()
    result = simplejson.loads(result)
    assert key == "uuid.by-dataset-metadata.json"
    assert result == expected
    result_from_dict = DatasetMetadata.load_from_dict(result, store).to_dict()
    assert result_from_dict == expected
