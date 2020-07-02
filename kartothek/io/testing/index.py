# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from toolz.dicttoolz import valmap

from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io.eager import store_dataframes_as_dataset


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
    dataset_factory = DatasetFactory(dataset_uuid, store_factory, load_all_indices=True)
    assert dataset_factory.indices.keys() == {"p", "x"}


def test_indices_uints(store_factory, metadata_version, bound_build_dataset_indices):
    dataset_uuid = "dataset_uuid"

    # min uint64
    p1 = 0

    # max uint64 => cannot even be cast to float32
    p2 = int(~np.uint64(0))

    # number that would be cut if converted to float64 and back
    p3 = 17128351978467489013

    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": pd.Series([p1], dtype=np.uint64)}))],
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": pd.Series([p2], dtype=np.uint64)}))],
        },
        {
            "label": "cluster_3",
            "data": [("core", pd.DataFrame({"p": pd.Series([p3], dtype=np.uint64)}))],
        },
    ]
    expected = {p1: ["cluster_1"], p2: ["cluster_2"], p3: ["cluster_3"]}

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
    assert_index_dct_equal(expected, dataset_factory.indices["p"].index_dct)

    # Re-create indices
    bound_build_dataset_indices(store_factory, dataset_uuid, columns=["p"])

    # Assert indices are properly created
    dataset_factory = DatasetFactory(dataset_uuid, store_factory, load_all_indices=True)
    assert_index_dct_equal(expected, dataset_factory.indices["p"].index_dct)


def test_empty_partitions(store_factory, metadata_version, bound_build_dataset_indices):
    dataset_uuid = "dataset_uuid"

    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": pd.Series([], dtype=np.int8)}))],
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": pd.Series([1], dtype=np.int8)}))],
        },
    ]
    expected = {1: ["cluster_2"]}

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
    assert_index_dct_equal(expected, dataset_factory.indices["p"].index_dct)
