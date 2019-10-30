import pandas as pd
import pytest

from kartothek.io_components.stats import _add_partition_col_to_stats_df, get_stats_df


@pytest.fixture
def dummy_df():
    columns = ["key", "num_rows", "num_row_group", "file_size"]
    data = [
        ("uuid/dataset_name/partition_name/part0", 1, 1, 1),
        ("uuid/dataset_name/partition_name/part1", 1, 1, 1),
        ("uuid/dataset_name/partition_name/part2", 1, 1, 1),
    ]
    return pd.DataFrame(data, columns=columns)


def test_get_stats_df_smoke(mocker):
    mocker.patch(
        "kartothek.io_components.stats._get_dataset_keys",
        return_value=set(
            [
                "uuid/dataset_name/partition_name/part0",
                "uuid/dataset_name/partition_name/part1",
                "uuid/dataset_name/partition_name/part2",
            ]
        ),
    )
    actual_df = get_stats_df(mocker.MagicMock, mocker.MagicMock)
    assert len(actual_df) == 3


def test_add_partition_col_to_stats_df(dummy_df):
    columns = ["key", "num_rows", "num_row_group", "file_size", "partition"]
    data = [
        ("uuid/dataset_name/partition_name/part0", 1, 1, 1, "partition_name"),
        ("uuid/dataset_name/partition_name/part1", 1, 1, 1, "partition_name"),
        ("uuid/dataset_name/partition_name/part2", 1, 1, 1, "partition_name"),
    ]
    expected_df = pd.DataFrame(data, columns=columns)
    actual_df = _add_partition_col_to_stats_df(dummy_df)
    assert expected_df.equals(actual_df)


def test_get_stats_df(dataset, store_session_factory):
    stats_df = get_stats_df("dataset_uuid", store_session_factory)
    assert len(stats_df) == 7
    assert stats_df.key.isin(
        [
            "dataset_uuid/helper/_common_metadata",
            "dataset_uuid.by-dataset-metadata.json",
            "dataset_uuid/helper/cluster_1.parquet",
        ]
    ).any()
