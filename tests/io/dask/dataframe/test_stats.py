import pandas as pd
import pytest

from kartothek.io.dask.dataframe import collect_dataset_statistics
from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.serialization import ParquetSerializer


def test_collect_dataset_statistics(store_session_factory, dataset):
    df_stats = collect_dataset_statistics(
        store=store_session_factory,
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


def test_collect_dataset_statistics_predicates(store_session_factory, dataset):
    predicates = [[("P", "==", 1)]]

    df_stats = collect_dataset_statistics(
        store=store_session_factory,
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


def test_collect_dataset_statistics_predicates_on_index(store_factory):
    df = pd.DataFrame(
        data={"P": range(10), "L": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]}
    )
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid="dataset_uuid", partition_on=["L"], dfs=[df],
    )
    predicates = [[("L", "==", "b")]]

    df_stats = collect_dataset_statistics(
        store=store_factory,
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


def test_collect_dataset_statistics_predicates_row_group_size(store_factory):
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

    df_stats = collect_dataset_statistics(
        store=store_factory,
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


def test_collect_dataset_statistics_frac_smoke(store_session_factory, dataset):
    df_stats = collect_dataset_statistics(
        store=store_session_factory,
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


def test_collect_dataset_statistics_frac_too_small(store_session_factory, dataset):
    with pytest.raises(ValueError):
        df_stats = collect_dataset_statistics(
            store=store_session_factory,
            dataset_uuid="dataset_uuid",
            table_name="table",
            frac=0.05,
        )
