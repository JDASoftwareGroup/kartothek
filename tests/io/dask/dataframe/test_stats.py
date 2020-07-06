# import pytest
import pandas as pd

from kartothek.io.dask.dataframe import collect_dataset_statistics


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
            "number_rows": [1, 1],
            "number_row_groups": [1, 1],
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

    expected = pd.DataFrame(
        data={
            "partition_label": ["cluster_1"],
            "row_group_id": [0],
            "number_rows": [1],
            "number_row_groups": [1],
        }
    )
    # TODO this fails, might be an issue with the predicate passing to the metapartitions in `dispatch_metapartitions`?
    pd.testing.assert_frame_equal(actual, expected)
