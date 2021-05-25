import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from kartothek.core.common_metadata import make_meta, read_schema_metadata
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex
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
        table_meta={
            "table1": pd.DataFrame({"col": [1]}),
            "table2": pd.DataFrame({"other_col": ["a"]}),
        },
        dataset_uuid="some_dataset",
        metadata_version=metadata_version,
    )

    new_data = {
        "data": {
            "table1": pd.DataFrame({"col": [1, 2]}),
            "table2": pd.DataFrame({"other_col": ["a", "b"]}),
        }
    }
    keys_in_store = set(store_factory().keys())
    new_mp = write_single_partition(
        store=store_factory, dataset_uuid="some_dataset", data=new_data
    )

    keys_in_store.add("some_dataset/table1/auto_dataset_uuid.parquet")
    keys_in_store.add("some_dataset/table2/auto_dataset_uuid.parquet")
    assert set(store_factory().keys()) == keys_in_store
    expected_mp = MetaPartition(
        label="auto_dataset_uuid",  # this will be a hash of the input
        files={
            "table1": "some_dataset/table1/auto_dataset_uuid.parquet",
            "table2": "some_dataset/table2/auto_dataset_uuid.parquet",
        },
        metadata_version=4,
        table_meta={
            "table1": make_meta(pd.DataFrame({"col": [1, 2]}), origin="table1"),
            "table2": make_meta(
                pd.DataFrame({"other_col": ["a", "b"]}), origin="table2"
            ),
        },
    )

    assert new_mp == expected_mp

    with pytest.raises(ValueError):
        # col is an integer column so this is incompatible.
        new_data["data"]["table1"] = pd.DataFrame({"col": [datetime.date(2010, 1, 1)]})
        write_single_partition(
            store=store_factory, dataset_uuid="some_dataset", data=new_data
        )


def test_create_dataset_header_minimal_version(store, metadata_storage_format):
    with pytest.raises(NotImplementedError):
        create_empty_dataset_header(
            store=store,
            table_meta={"table": pd.DataFrame({"col": [1]})},
            dataset_uuid="new_dataset_uuid",
            metadata_storage_format=metadata_storage_format,
            metadata_version=3,
        )
    create_empty_dataset_header(
        store=store,
        table_meta={"table": pd.DataFrame({"col": [1]})},
        dataset_uuid="new_dataset_uuid",
        metadata_storage_format=metadata_storage_format,
        metadata_version=4,
    )


def test_create_dataset_header(store, metadata_storage_format, frozen_time):
    table_meta = {"table": make_meta(pd.DataFrame({"col": [1]}), origin="1")}
    new_dataset = create_empty_dataset_header(
        store=store,
        table_meta=table_meta,
        dataset_uuid="new_dataset_uuid",
        metadata_storage_format=metadata_storage_format,
        metadata_version=4,
    )

    expected_dataset = DatasetMetadata(
        uuid="new_dataset_uuid",
        metadata_version=4,
        explicit_partitions=False,
        table_meta=table_meta,
    )
    assert new_dataset == expected_dataset

    storage_keys = list(store.keys())
    assert len(storage_keys) == 2

    loaded = DatasetMetadata.load_from_store(store=store, uuid="new_dataset_uuid")
    assert loaded == expected_dataset

    # If the read succeeds, the schema is written
    read_schema_metadata(dataset_uuid=new_dataset.uuid, store=store, table="table")


# TODO: move `store_dataframes_as_dataset` tests to generic tests or remove if redundant
def test_store_dataframes_as_dataset_no_pipeline_partition_on(store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df2 = pd.DataFrame({"P": np.arange(0, 10), "info": np.arange(100, 110)})

    dataset = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[{"core": df, "helper": df2}],
        partition_on="P",
        metadata_version=4,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 10

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


@pytest.mark.parametrize("test_input", ["NOT_IN_COLUMNS"])
def test_store_dataframes_as_dataset_partition_on_inconsistent(test_input, store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df2 = pd.DataFrame({"P": np.arange(0, 10), "info": np.arange(100, 110)})

    with pytest.raises(ValueError) as excinfo:
        store_dataframes_as_dataset(
            store=store,
            dataset_uuid="dataset_uuid",
            dfs=[{"core": df, "helper": df2}],
            partition_on=[test_input],
            metadata_version=4,
        )
    assert str(excinfo.value) == "Partition column(s) missing: {}".format(test_input)


def test_store_dataframes_as_dataset_no_pipeline(metadata_version, store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df2 = pd.DataFrame({"P": np.arange(0, 10), "info": np.arange(100, 110)})

    dataset = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[{"core": df, "helper": df2}],
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 1
    assert dataset.metadata_version == metadata_version

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


def test_store_dataframes_as_dataset_dfs_input_formats(store):
    df1 = pd.DataFrame({"B": [pd.Timestamp("2019")]})
    df2 = pd.DataFrame({"A": [1.4]})
    formats = [
        {"data": {"D": df1, "S": df2}},
        {"D": df1, "S": df2},
        {"data": [("D", df1), ("S", df2)]},
        [("D", df1), ("S", df2)],
    ]
    for input_format in formats:
        dataset = store_dataframes_as_dataset(
            store=store, dataset_uuid="dataset_uuid", dfs=[input_format], overwrite=True
        )
        stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
        assert dataset == stored_dataset


def test_store_dataframes_as_dataset_mp(metadata_version, store):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df2 = pd.DataFrame({"P": np.arange(0, 10), "info": np.arange(100, 110)})

    mp = MetaPartition(
        label=gen_uuid(),
        data={"core": df, "helper": df2},
        metadata_version=metadata_version,
    )

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

    input_ = [
        {
            "label": "label",
            "data": [("order_proposals", df)],
            "indices": {"location": {k: ["label"] for k in df["location"].unique()}},
        }
    ]
    dataset = store_dataframes_as_dataset(
        dfs=input_,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=4,
        partition_on=["other"],
    )

    new_data = {
        "data": {
            "order_proposals": pd.DataFrame(
                OrderedDict([("other", ["b", "b", "b"]), ("location", ["0", "1", "2"])])
            )
        }
    }
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

    new_data["label"] = "some_other_label"
    # If no partitioning is given, it will be determined based on the existing dataset
    write_single_partition(
        store=store_factory, dataset_uuid=dataset.uuid, data=new_data
    )
    assert initial_keys + 2 == len(list(store_factory().keys()))


def test_store_dataframes_as_dataset_does_not_allow_invalid_indices(store_factory):
    partitions = [
        {
            "label": "part1",
            "data": [("core", pd.DataFrame({"p": [1, 2]}))],
            "indices": {"x": ExplicitSecondaryIndex("x", {1: ["part1"], 2: ["part2"]})},
        }
    ]

    with pytest.raises(
        ValueError, match="In table core, no column corresponding to index x"
    ):
        store_dataframes_as_dataset(
            dfs=partitions,
            store=store_factory,
            metadata={"dataset": "metadata"},
            dataset_uuid="dataset_uuid",
        )
