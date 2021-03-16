import pandas as pd
import pytest

from kartothek.io.dask.dataframe import collect_dataset_metadata
from kartothek.io.eager import (
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
)
from kartothek.io_components.metapartition import _METADATA_SCHEMA
from kartothek.serialization import ParquetSerializer


def test_collect_dataset_metadata(store_session_factory, dataset):
    df_stats = collect_dataset_metadata(
        store=store_session_factory,
        dataset_uuid="dataset_uuid",
        predicates=None,
        frac=1,
    ).compute()

    actual = df_stats.drop(
        columns=[
            "row_group_compressed_size",
            "row_group_uncompressed_size",
            "serialized_size",
        ],
        axis=1,
    )
    actual.sort_values(by=["partition_label", "row_group_id"], inplace=True)

    expected = pd.DataFrame(
        data={
            "partition_label": ["cluster_1", "cluster_2"],
            "row_group_id": [0, 0],
            "number_rows_total": [1, 1],
            "number_row_groups": [1, 1],
            "number_rows_per_row_group": [1, 1],
        },
        index=[0, 0],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_predicates(store_session_factory, dataset):
    predicates = [[("P", "==", 1)]]

    df_stats = collect_dataset_metadata(
        store=store_session_factory,
        dataset_uuid="dataset_uuid",
        predicates=predicates,
        frac=1,
    ).compute()

    actual = df_stats.drop(
        columns=[
            "row_group_compressed_size",
            "row_group_uncompressed_size",
            "serialized_size",
        ],
        axis=1,
    )
    actual.sort_values(by=["partition_label", "row_group_id"], inplace=True)

    # Predicates are only evaluated on index level and have therefore no effect on this dataset
    expected = pd.DataFrame(
        data={
            "partition_label": ["cluster_1", "cluster_2"],
            "row_group_id": [0, 0],
            "number_rows_total": [1, 1],
            "number_row_groups": [1, 1],
            "number_rows_per_row_group": [1, 1],
        },
        index=[0, 0],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_predicates_on_index(store_factory):
    df = pd.DataFrame(
        data={"P": range(10), "L": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]}
    )
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", partition_on=["L"], dfs=[df],
    )
    predicates = [[("L", "==", "b")]]

    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid", predicates=predicates, frac=1,
    ).compute()

    assert "L=b" in df_stats["partition_label"].values[0]

    df_stats.sort_values(by=["partition_label", "row_group_id"], inplace=True)
    actual = df_stats.drop(
        columns=[
            "partition_label",
            "row_group_compressed_size",
            "row_group_uncompressed_size",
            "serialized_size",
        ],
        axis=1,
    )

    expected = pd.DataFrame(
        data={
            "row_group_id": [0],
            "number_rows_total": [5],
            "number_row_groups": [1],
            "number_rows_per_row_group": [5],
        },
        index=[0],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_predicates_row_group_size(store_factory):
    ps = ParquetSerializer(chunk_size=2)
    df = pd.DataFrame(
        data={"P": range(10), "L": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]}
    )
    store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid="dataset_uuid",
        partition_on=["L"],
        dfs=[df],
        df_serializer=ps,
    )

    predicates = [[("L", "==", "a")]]

    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid", predicates=predicates, frac=1,
    ).compute()

    for part_label in df_stats["partition_label"]:
        assert "L=a" in part_label
    df_stats.sort_values(by=["partition_label", "row_group_id"], inplace=True)

    actual = df_stats.drop(
        columns=[
            "partition_label",
            "row_group_compressed_size",
            "row_group_uncompressed_size",
            "serialized_size",
        ],
        axis=1,
    )

    expected = pd.DataFrame(
        data={
            "row_group_id": [0, 1, 2],
            "number_rows_total": [5, 5, 5],
            "number_row_groups": [3, 3, 3],
            "number_rows_per_row_group": [2, 2, 1],
        },
        index=[0, 1, 2],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_frac_smoke(store_session_factory, dataset):
    df_stats = collect_dataset_metadata(
        store=store_session_factory, dataset_uuid="dataset_uuid", frac=0.8,
    ).compute()
    columns = {
        "partition_label",
        "row_group_id",
        "row_group_compressed_size",
        "row_group_uncompressed_size",
        "number_rows_total",
        "number_row_groups",
        "serialized_size",
        "number_rows_per_row_group",
    }

    assert set(df_stats.columns) == columns


def test_collect_dataset_metadata_empty_dataset(store_factory):
    df = pd.DataFrame(columns=["A", "b"], index=pd.RangeIndex(start=0, stop=0))
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"]
    )
    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid"
    ).compute()
    expected = pd.DataFrame(columns=_METADATA_SCHEMA.keys())
    expected = expected.astype(_METADATA_SCHEMA)
    pd.testing.assert_frame_equal(expected, df_stats)


def test_collect_dataset_metadata_concat(store_factory):
    """Smoke-test concatenation of empty and non-empty dataset metadata collections."""
    df = pd.DataFrame(data={"A": [1, 1, 1, 1], "b": [1, 1, 2, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"]
    )
    df_stats1 = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid",
    ).compute()

    # Remove all partitions of the dataset
    update_dataset_from_dataframes(
        [], store=store_factory, dataset_uuid="dataset_uuid", delete_scope=[{"A": 1}]
    )

    df_stats2 = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid",
    ).compute()
    pd.concat([df_stats1, df_stats2])


def test_collect_dataset_metadata_delete_dataset(store_factory):
    df = pd.DataFrame(data={"A": [1, 1, 1, 1], "b": [1, 1, 2, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"]
    )
    # Remove all partitions of the dataset
    update_dataset_from_dataframes(
        [], store=store_factory, dataset_uuid="dataset_uuid", delete_scope=[{"A": 1}]
    )

    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid",
    ).compute()
    expected = pd.DataFrame(columns=_METADATA_SCHEMA)
    expected = expected.astype(_METADATA_SCHEMA)
    pd.testing.assert_frame_equal(expected, df_stats)


def test_collect_dataset_metadata_fraction_precision(store_factory):
    df = pd.DataFrame(data={"A": range(100), "B": range(100)})

    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"],
    )  # Creates 100 partitions

    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid", frac=0.2
    ).compute()
    assert len(df_stats) == 20


def test_collect_dataset_metadata_at_least_one_partition(store_factory):
    """
    Make sure we return at leat one partition, even if none would be returned by rounding frac * n_partitions
    """
    df = pd.DataFrame(data={"A": range(100), "B": range(100)})

    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"],
    )  # Creates 100 partitions

    df_stats = collect_dataset_metadata(
        store=store_factory, dataset_uuid="dataset_uuid", frac=0.005
    ).compute()
    assert len(df_stats) == 1


def test_collect_dataset_metadata_invalid_frac(store_session_factory, dataset):
    with pytest.raises(ValueError, match="Invalid value for parameter `frac`"):
        collect_dataset_metadata(
            store=store_session_factory, dataset_uuid="dataset_uuid", frac=1.1,
        )

    with pytest.raises(ValueError, match="Invalid value for parameter `frac`"):
        collect_dataset_metadata(
            store=store_session_factory, dataset_uuid="dataset_uuid", frac=0.0,
        )
