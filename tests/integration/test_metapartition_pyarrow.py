# -*- coding: utf-8 -*-


import pandas as pd

from kartothek.io_components.metapartition import MetaPartition


def test_eq():
    df = pd.DataFrame({"a": [1]})
    df_same = pd.DataFrame({"a": [1]})
    df_other = pd.DataFrame({"a": [2]})

    meta_partition = MetaPartition.from_dict(
        {"label": "test_label", "data": {"core": df}}
    )
    assert meta_partition == meta_partition

    meta_partition_same = MetaPartition.from_dict(
        {"label": "test_label", "data": {"core": df_same}}
    )
    assert meta_partition == meta_partition_same

    meta_partition_different_df = MetaPartition.from_dict(
        {"label": "test_label", "data": {"core": df_other}}
    )
    assert not meta_partition == meta_partition_different_df

    meta_partition_different_label = MetaPartition.from_dict(
        {"label": "test_label", "data": {"not_core": df_same}}
    )
    assert not meta_partition == meta_partition_different_label

    meta_partition_empty_data = MetaPartition.from_dict(
        {"label": "test_label", "data": {}}
    )
    assert meta_partition_empty_data == meta_partition_empty_data

    meta_partition_more_data = MetaPartition.from_dict(
        {"label": "test_label", "data": {"core": df, "not_core": df}}
    )
    assert not (meta_partition == meta_partition_more_data)

    assert not meta_partition == "abc"
