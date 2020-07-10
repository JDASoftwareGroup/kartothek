import numpy as np
import pandas as pd
import pytest

from kartothek.io.dask.dataframe import collect_dataset_metadata
from kartothek.io.eager import (
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
)
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import store_dataset_from_partitions
from kartothek.serialization import ParquetSerializer

METADATA_COLUMNS = (
    "partition_label",
    "row_group_id",
    "row_group_byte_size",
    "number_rows_total",
    "number_row_groups",
    "serialized_size",
    "number_rows_per_row_group",
)

METADATA_DTYPES = (
    np.dtype("O"),
    np.dtype(int),
    np.dtype(int),
    np.dtype(int),
    np.dtype(int),
    np.dtype(int),
    np.dtype(int),
)


def test_collect_dataset_metadata(store_session_factory, dataset):
    df_stats = collect_dataset_metadata(
        store_factory=store_session_factory,
        dataset_uuid="dataset_uuid",
        table_name="table",
        predicates=None,
        frac=1,
    )
    actual = df_stats.drop(columns=["row_group_byte_size", "serialized_size"], axis=1)

    expected = pd.DataFrame(
        data={
            "partition_label": ["cluster_1", "cluster_2"],
            "row_group_id": [0, 0],
            "number_rows_total": [1, 1],
            "number_row_groups": [1, 1],
            "number_rows_per_row_group": [1, 1],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_predicates(store_session_factory, dataset):
    predicates = [[("P", "==", 1)]]

    df_stats = collect_dataset_metadata(
        store_factory=store_session_factory,
        dataset_uuid="dataset_uuid",
        table_name="table",
        predicates=predicates,
        frac=1,
    )
    actual = df_stats.drop(columns=["row_group_byte_size", "serialized_size"], axis=1)

    # Predicates are only evaluated on index level and have therefore no effect on this dataset
    expected = pd.DataFrame(
        data={
            "partition_label": ["cluster_1", "cluster_2"],
            "row_group_id": [0, 0],
            "number_rows_total": [1, 1],
            "number_row_groups": [1, 1],
            "number_rows_per_row_group": [1, 1],
        }
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
        store_factory=store_factory,
        dataset_uuid="dataset_uuid",
        table_name="table",
        predicates=predicates,
        frac=1,
    )
    assert "L=b" in df_stats["partition_label"].values[0]
    actual = df_stats.drop(
        columns=["partition_label", "row_group_byte_size", "serialized_size"], axis=1
    )

    expected = pd.DataFrame(
        data={
            "row_group_id": [0],
            "number_rows_total": [5],
            "number_row_groups": [1],
            "number_rows_per_row_group": [5],
        }
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
        store_factory=store_factory,
        dataset_uuid="dataset_uuid",
        table_name="table",
        predicates=predicates,
        frac=1,
    )
    for part_label in df_stats["partition_label"]:
        assert "L=a" in part_label

    actual = df_stats.drop(
        columns=["partition_label", "row_group_byte_size", "serialized_size"], axis=1
    )

    expected = pd.DataFrame(
        data={
            "row_group_id": [0, 1, 2],
            "number_rows_total": [5, 5, 5],
            "number_row_groups": [3, 3, 3],
            "number_rows_per_row_group": [2, 2, 1],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_collect_dataset_metadata_frac_smoke(store_session_factory, dataset):
    df_stats = collect_dataset_metadata(
        store_factory=store_session_factory,
        dataset_uuid="dataset_uuid",
        table_name="table",
        frac=0.8,
    )
    columns = {
        "partition_label",
        "row_group_id",
        "row_group_byte_size",
        "number_rows_total",
        "number_row_groups",
        "serialized_size",
        "number_rows_per_row_group",
    }

    assert set(df_stats.columns) == columns


def test_collect_dataset_metadata_empty_dataset_mp(store_factory):
    mp = MetaPartition(label="cluster_1")
    store_dataset_from_partitions(
        partition_list=[mp], store=store_factory, dataset_uuid="dataset_uuid"
    )

    df_stats = collect_dataset_metadata(
        store_factory=store_factory, dataset_uuid="dataset_uuid", table_name="table"
    )
    expected = pd.DataFrame(columns=METADATA_COLUMNS)
    expected = expected.astype(dict(zip(METADATA_COLUMNS, METADATA_DTYPES)))
    pd.testing.assert_frame_equal(expected, df_stats, check_index_type=False)


def test_collect_dataset_metadata_empty_dataset(store_factory):
    df = pd.DataFrame(columns=["A", "b"], index=pd.RangeIndex(start=0, stop=0))
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"]
    )
    with pytest.warns(
        UserWarning, match="^Can't retrieve metadata for empty dataset.*"
    ):
        df_stats = collect_dataset_metadata(
            store_factory=store_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
        )
    expected = pd.DataFrame(columns=METADATA_COLUMNS)
    expected = expected.astype(dict(zip(METADATA_COLUMNS, METADATA_DTYPES)))
    pd.testing.assert_frame_equal(expected, df_stats)


def test_collect_dataset_metadata_concat(store_factory):
    """Smoke-test concatenation of empty and non-empty dataset metadata collections."""
    df = pd.DataFrame(data={"A": [1, 1, 1, 1], "b": [1, 1, 2, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", dfs=[df], partition_on=["A"]
    )
    df_stats1 = collect_dataset_metadata(
        store_factory=store_factory, dataset_uuid="dataset_uuid", table_name="table",
    )

    # Remove all partitions of the dataset
    update_dataset_from_dataframes(
        [], store=store_factory, dataset_uuid="dataset_uuid", delete_scope=[{"A": 1}]
    )

    with pytest.warns(
        UserWarning, match="^Can't retrieve metadata for empty dataset.*"
    ):
        df_stats2 = collect_dataset_metadata(
            store_factory=store_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
        )
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

    with pytest.warns(
        UserWarning, match="^Can't retrieve metadata for empty dataset.*"
    ):
        df_stats = collect_dataset_metadata(
            store_factory=store_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
        )
    expected = pd.DataFrame(columns=METADATA_COLUMNS)
    expected = expected.astype(dict(zip(METADATA_COLUMNS, METADATA_DTYPES)))
    pd.testing.assert_frame_equal(expected, df_stats)


def test_collect_dataset_metadata_table_without_partition(store_factory):
    """
    df2 doesn't have files for all partition (specificaly `A==2`).
    Make sure that we still collect the right metadata
    """
    df1 = pd.DataFrame(data={"A": [1, 1, 2, 2], "b": [1, 1, 2, 2]})
    df2 = pd.DataFrame(data={"A": [1, 1], "b": [1, 1]})

    store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid="dataset_uuid",
        dfs=[{"table1": df1, "table2": df2}],
        partition_on=["A"],
    )

    # df_stats = collect_dataset_metadata(
    #     store_factory=store_factory, dataset_uuid="dataset_uuid", table_name="table2",
    # )
    # TODO assertions


def test_collect_dataset_metadata_invalid_frac(store_session_factory, dataset):
    with pytest.raises(ValueError, match="Invalid value for parameter `frac`"):
        collect_dataset_metadata(
            store_factory=store_session_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
            frac=1.1,
        )

    with pytest.raises(ValueError, match="Invalid value for parameter `frac`"):
        collect_dataset_metadata(
            store_factory=store_session_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
            frac=0.0,
        )
