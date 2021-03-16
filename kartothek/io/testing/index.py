# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from toolz.dicttoolz import valmap

from kartothek.core.factory import DatasetFactory
from kartothek.io.eager import store_dataframes_as_dataset


def assert_index_dct_equal(dict1, dict2):
    dict1 = valmap(sorted, dict1)
    dict2 = valmap(sorted, dict2)
    assert dict1 == dict2


def test_build_indices(store_factory, metadata_version, bound_build_dataset_indices):
    dataset_uuid = "dataset_uuid"
    partitions = [
        pd.DataFrame({"p": [1, 2]}),
        pd.DataFrame({"p": [2, 3]}),
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
    index_dct = dataset_factory.indices["p"].index_dct

    assert len(index_dct[1]) == 1
    assert len(index_dct[2]) == 2
    assert len(index_dct[3]) == 1

    assert len(set(index_dct[3]) & set(index_dct[2])) == 1
    assert len(set(index_dct[1]) & set(index_dct[2])) == 1
    assert len(set(index_dct[1]) & set(index_dct[3])) == 0


def test_create_index_from_inexistent_column_fails(
    store_factory, metadata_version, bound_build_dataset_indices
):
    dataset_uuid = "dataset_uuid"
    partitions = [
        pd.DataFrame({"p": [1, 2]}),
        pd.DataFrame({"p": [2, 3]}),
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
        pd.DataFrame({"p": [1, 2], "x": [100, 4500]}),
        pd.DataFrame({"p": [4, 3], "x": [500, 10]}),
    ]

    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        secondary_indices="p",
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
        pd.DataFrame({"p": pd.Series([p1], dtype=np.uint64)}),
        pd.DataFrame({"p": pd.Series([p2], dtype=np.uint64)}),
        pd.DataFrame({"p": pd.Series([p3], dtype=np.uint64)}),
    ]

    def assert_expected(index_dct):
        assert len(index_dct) == 3
        referenced_partitions = []
        for val in index_dct.values():
            referenced_partitions.extend(val)
        assert len(referenced_partitions) == 3

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
    assert_expected(dataset_factory.indices["p"].index_dct)
    first_run = dataset_factory.indices["p"].index_dct.copy()

    # Re-create indices
    bound_build_dataset_indices(store_factory, dataset_uuid, columns=["p"])

    # Assert indices are properly created
    dataset_factory = DatasetFactory(dataset_uuid, store_factory, load_all_indices=True)
    assert_index_dct_equal(first_run, dataset_factory.indices["p"].index_dct)


def test_empty_partitions(store_factory, metadata_version, bound_build_dataset_indices):
    dataset_uuid = "dataset_uuid"

    partitions = [
        pd.DataFrame({"p": pd.Series([], dtype=np.int8)}),
        pd.DataFrame({"p": pd.Series([1], dtype=np.int8)}),
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
    assert len(dataset_factory.indices["p"].index_dct) == 1
