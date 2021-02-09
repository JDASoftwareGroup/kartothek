import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from kartothek.core.common_metadata import make_meta, read_schema_metadata
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import (
    create_empty_dataset_header,
    store_dataframes_as_dataset,
    write_single_partition,
)
from kartothek.io.testing.write import *  # noqa: F40
from kartothek.io_components.metapartition import MetaPartition


def _store_dataframes(dfs, **kwargs):
    # Positional arguments in function but `None` is acceptable input
    for kw in ("dataset_uuid", "store"):
        if kw not in kwargs:
            kwargs[kw] = None

    return store_dataframes_as_dataset(dfs=dfs, **kwargs)


@pytest.fixture()
def bound_store_dataframes():
    return _store_dataframes


def test_write_single_partition(store_factory, mock_uuid, metadata_version):
    create_empty_dataset_header(
        store=store_factory(),
        schema=pd.DataFrame({"col": [1]}),
        dataset_uuid="some_dataset",
        metadata_version=metadata_version,
        table_name="table1",
    )

    new_data = pd.DataFrame({"col": [1, 2]})
    keys_in_store = set(store_factory().keys())
    new_mp = write_single_partition(
        store=store_factory,
        dataset_uuid="some_dataset",
        data=new_data,
        table_name="table1",
    )

    keys_in_store.add("some_dataset/table1/auto_dataset_uuid.parquet")
    assert set(store_factory().keys()) == keys_in_store
    expected_mp = MetaPartition(
        label="auto_dataset_uuid",  # this will be a hash of the input
        file="some_dataset/table1/auto_dataset_uuid.parquet",
        metadata_version=4,
        schema=make_meta(pd.DataFrame({"col": [1, 2]}), origin="table1"),
    )

    assert new_mp == expected_mp

    with pytest.raises(ValueError):
        # col is an integer column so this is incompatible.
        new_data = pd.DataFrame({"col": [datetime.date(2010, 1, 1)]})
        write_single_partition(
            store=store_factory,
            dataset_uuid="some_dataset",
            data=new_data,
            table_name="table1",
        )


def test_create_dataset_header_minimal_version(store, metadata_storage_format):
    with pytest.raises(NotImplementedError):
        create_empty_dataset_header(
            store=store,
            schema=pd.DataFrame({"col": [1]}),
            dataset_uuid="new_dataset_uuid",
            metadata_storage_format=metadata_storage_format,
            metadata_version=3,
        )
    create_empty_dataset_header(
        store=store,
        schema=pd.DataFrame({"col": [1]}),
        dataset_uuid="new_dataset_uuid",
        metadata_storage_format=metadata_storage_format,
        metadata_version=4,
    )


def test_create_dataset_header(store, metadata_storage_format, frozen_time):
    schema = make_meta(pd.DataFrame({"col": [1]}), origin="1")
    new_dataset = create_empty_dataset_header(
        store=store,
        schema=schema,
        dataset_uuid="new_dataset_uuid",
        metadata_storage_format=metadata_storage_format,
        metadata_version=4,
    )

    expected_dataset = DatasetMetadata(
        uuid="new_dataset_uuid",
        metadata_version=4,
        explicit_partitions=False,
        schema=schema,
    )
    assert new_dataset == expected_dataset

    storage_keys = list(store.keys())
    assert len(storage_keys) == 2

    loaded = DatasetMetadata.load_from_store(store=store, uuid="new_dataset_uuid")
    assert loaded == expected_dataset

    # If the read succeeds, the schema is written
    read_schema_metadata(dataset_uuid=new_dataset.uuid, store=store)


# TODO: move `store_dataframes_as_dataset` tests to generic tests or remove if redundant
def test_store_dataframes_as_dataset_no_pipeline_partition_on(store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    dataset = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[df],
        partition_on="P",
        metadata_version=4,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 10

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


def test_store_dataframes_as_dataset_partition_on_inconsistent(store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    with pytest.raises(ValueError) as excinfo:
        store_dataframes_as_dataset(
            store=store,
            dataset_uuid="dataset_uuid",
            dfs=[df],
            partition_on=["NOT_IN_COLUMNS"],
            metadata_version=4,
        )
    assert str(excinfo.value) == "Partition column(s) missing: NOT_IN_COLUMNS"


def test_store_dataframes_as_dataset_no_pipeline(metadata_version, store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    dataset = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[df],
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 1
    assert dataset.metadata_version == metadata_version

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


def test_store_dataframes_as_dataset_mp(metadata_version, store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    mp = MetaPartition(label=gen_uuid(), data=df, metadata_version=metadata_version,)

    dataset = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[mp],
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 1
    assert dataset.metadata_version == metadata_version

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


def test_write_single_partition_different_partitioning(store_factory):
    df = pd.DataFrame(
        OrderedDict([("location", ["0", "1", "2"]), ("other", ["a", "a", "a"])])
    )

    input_ = [df]

    dataset = store_dataframes_as_dataset(
        dfs=input_,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=4,
        partition_on=["other"],
    )

    new_data = [
        pd.DataFrame(
            OrderedDict([("other", ["b", "b", "b"]), ("location", ["0", "1", "2"])])
        )
    ]
    initial_keys = len(list(store_factory().keys()))
    with pytest.raises(ValueError):
        write_single_partition(
            store=store_factory,
            dataset_uuid=dataset.uuid,
            data=new_data,
            partition_on="location",
        )
    assert initial_keys == len(list(store_factory().keys()))
    write_single_partition(
        store=store_factory,
        dataset_uuid=dataset.uuid,
        data=new_data,
        partition_on=["other"],
    )
    assert initial_keys + 1 == len(list(store_factory().keys()))

    # If no partitioning is given, it will be determined based on the existing dataset
    write_single_partition(
        store=store_factory, dataset_uuid=dataset.uuid, data=new_data
    )
    assert initial_keys + 2 == len(list(store_factory().keys()))
