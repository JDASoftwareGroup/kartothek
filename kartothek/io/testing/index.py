# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from toolz.dicttoolz import valmap

from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io.eager import (
    read_dataset_as_metapartitions,
    store_dataframes_as_dataset,
)


def assert_index_dct_equal(dict1, dict2):
    dict1 = valmap(sorted, dict1)
    dict2 = valmap(sorted, dict2)
    assert dict1 == dict2


def test_build_indices(store_factory, metadata_version, bound_build_dataset_indices):
    dataset_uuid = "dataset_uuid"
    partitions = [
        {"label": "cluster_1", "data": [("core", pd.DataFrame({"p": [1, 2]}))]},
        {"label": "cluster_2", "data": [("core", pd.DataFrame({"p": [2, 3]}))]},
    ]

    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
    )
    dataset = dataset.load_all_indices(store=store_factory)
    assert not dataset.indices

    # Create indices
    bound_build_dataset_indices(store_factory, dataset_uuid, columns=["p"])

    # Assert indices are properly created
    dataset_factory = DatasetFactory(dataset_uuid, store_factory, load_all_indices=True)
    expected = {2: ["cluster_1", "cluster_2"], 3: ["cluster_2"], 1: ["cluster_1"]}
    assert_index_dct_equal(expected, dataset_factory.indices["p"].index_dct)


def test_create_index_from_inexistent_column_fails(
    store_factory, metadata_version, bound_build_dataset_indices
):
    dataset_uuid = "dataset_uuid"
    partitions = [
        {"label": "cluster_1", "data": [("core", pd.DataFrame({"p": [1, 2]}))]},
        {"label": "cluster_2", "data": [("core", pd.DataFrame({"p": [2, 3]}))]},
    ]

    store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
    )

    with pytest.raises(RuntimeError, match="Column `.*` could not be found"):
        bound_build_dataset_indices(store_factory, dataset_uuid, columns=["abc"])


def test_add_column_to_existing_index(
    store_factory, metadata_version, bound_build_dataset_indices
):
    dataset_uuid = "dataset_uuid"
    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": [1, 2], "x": [100, 4500]}))],
            "indices": {
                "p": ExplicitSecondaryIndex(
                    "p", index_dct={1: ["cluster_1"], 2: ["cluster_1"]}
                )
            },
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": [4, 3], "x": [500, 10]}))],
            "indices": {
                "p": ExplicitSecondaryIndex(
                    "p", index_dct={4: ["cluster_2"], 3: ["cluster_2"]}
                )
            },
        },
    ]

    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
    )
    assert dataset.load_all_indices(store=store_factory()).indices.keys() == {"p"}

    # Create indices
    bound_build_dataset_indices(store_factory, dataset_uuid, columns=["x"])

    # Assert indices are properly created
    mps = read_dataset_as_metapartitions(store=store_factory, dataset_uuid=dataset_uuid)
    for column_name in ["p", "x"]:
        assert all([mp.indices[column_name] for mp in mps])

    dataset_factory = DatasetFactory(dataset_uuid, store_factory, load_all_indices=True)
    assert dataset_factory.indices.keys() == {"p", "x"}
