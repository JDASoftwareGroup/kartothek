# -*- coding: utf-8 -*-
# pylint: disable=E1101


import pandas as pd
import pytest

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.testing import TIME_TO_FREEZE_ISO
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)


def test_store_dataset_from_partitions(meta_partitions_files_only, store, frozen_time):
    dataset = store_dataset_from_partitions(
        partition_list=meta_partitions_files_only,
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"some": "metadata"},
    )

    expected_metadata = {"some": "metadata", "creation_time": TIME_TO_FREEZE_ISO}

    assert dataset.metadata == expected_metadata
    assert sorted(dataset.partitions.values(), key=lambda x: x.label) == sorted(
        [mp.partition for mp in meta_partitions_files_only], key=lambda x: x.label
    )
    assert dataset.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # Dataset metadata: 1 file
    expected_number_files = 1
    # common metadata for v4 datasets
    expected_number_files += 2
    assert len(store_files) == expected_number_files

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset == stored_dataset


def test_store_dataset_from_partitions_update(store, metadata_version, frozen_time):
    mp1 = MetaPartition(
        label="cluster_1",
        data={"df": pd.DataFrame({"p": [1]})},
        files={"df": "1.parquet"},
        indices={"p": ExplicitSecondaryIndex("p", index_dct={1: ["cluster_1"]})},
        metadata_version=metadata_version,
    )
    mp2 = MetaPartition(
        label="cluster_2",
        data={"df": pd.DataFrame({"p": [2]})},
        files={"df": "2.parquet"},
        indices={"p": ExplicitSecondaryIndex("p", index_dct={2: ["cluster_2"]})},
        metadata_version=metadata_version,
    )
    dataset = store_dataset_from_partitions(
        partition_list=[mp1, mp2],
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"dataset": "metadata"},
    )
    dataset = dataset.load_index("p", store)

    mp3 = MetaPartition(
        label="cluster_3",
        data={"df": pd.DataFrame({"p": [3]})},
        files={"df": "3.parquet"},
        indices={"p": ExplicitSecondaryIndex("p", index_dct={3: ["cluster_3"]})},
        metadata_version=metadata_version,
    )

    dataset_updated = store_dataset_from_partitions(
        partition_list=[mp3],
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"extra": "metadata"},
        update_dataset=dataset,
        remove_partitions=["cluster_1"],
    )
    dataset_updated = dataset_updated.load_index("p", store)
    expected_metadata = {"dataset": "metadata", "extra": "metadata"}

    expected_metadata["creation_time"] = TIME_TO_FREEZE_ISO

    assert dataset_updated.metadata == expected_metadata
    assert list(dataset.partitions) == ["cluster_1", "cluster_2"]
    assert list(dataset_updated.partitions) == ["cluster_2", "cluster_3"]
    assert dataset_updated.partitions["cluster_3"] == mp3.partition
    assert dataset_updated.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # 1 dataset metadata file and 1 index file
    # note: the update writes a new index file but due to frozen_time this gets
    # the same name as the previous one and overwrites it.
    expected_number_files = 2
    # common metadata for v4 datasets (1 table)
    expected_number_files += 1
    assert len(store_files) == expected_number_files

    assert dataset.indices["p"].index_dct == {1: ["cluster_1"], 2: ["cluster_2"]}
    assert dataset_updated.indices["p"].index_dct == {
        2: ["cluster_2"],
        3: ["cluster_3"],
    }

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    stored_dataset = stored_dataset.load_index("p", store)
    assert dataset_updated == stored_dataset


def test_raise_if_dataset_exists(store_factory, dataset_function):
    raise_if_dataset_exists(dataset_uuid="ThisDoesNotExist", store=store_factory)
    with pytest.raises(RuntimeError):
        raise_if_dataset_exists(dataset_uuid=dataset_function.uuid, store=store_factory)
