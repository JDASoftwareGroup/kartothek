import types

import pandas as pd
import pytest

from kartothek.io_components.merge import align_datasets
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import store_dataset_from_partitions


def test_align_datasets_prefix(dataset, evaluation_dataset, store_session):
    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=evaluation_dataset.uuid,
        store=store_session,
        match_how="prefix",
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    # Two separate cluster_groups (e.g. cluster_1*)
    assert len(list_metapartitions) == 2

    mp_list = list_metapartitions[0]

    assert len(mp_list) == 3, [mp.label for mp in mp_list]

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 3, [mp.label for mp in mp_list]

    # Test sorting of datasets by length, i.e. order of dataframes is different
    generator = align_datasets(
        left_dataset_uuid=evaluation_dataset.uuid,
        right_dataset_uuid=dataset.uuid,
        store=store_session,
        match_how="prefix",
    )
    list_metapartitions = list(generator)
    mp_list = list_metapartitions[0]


def test_align_datasets_prefix__equal_number_of_partitions(
    dataset, evaluation_dataset, store_session
):
    """
    Test a scenario where the simple prefix match algorithm didn't find any
    matches in case of equal number of partitions in both datasets.
    """

    # Create a reference dataset which matches the problem (equal number of
    # partitions and suitable for prefix matching)
    mp = MetaPartition(label="cluster_1_1", metadata_version=dataset.metadata_version)
    mp2 = MetaPartition(label="cluster_2_1", metadata_version=dataset.metadata_version)
    metapartitions = [mp, mp2]
    store_dataset_from_partitions(
        partition_list=metapartitions,
        dataset_uuid="reference_dataset_uuid",
        store=store_session,
    )

    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid="reference_dataset_uuid",
        store=store_session,
        match_how="prefix",
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    # Two separate cluster_groups (e.g. cluster_1*)
    assert len(list_metapartitions) == 2

    mp_list = list_metapartitions[0]

    assert len(mp_list) == 2

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 2

    # Test sorting of datasets by length, i.e. order of dataframes is different
    generator = align_datasets(
        left_dataset_uuid=evaluation_dataset.uuid,
        right_dataset_uuid=dataset.uuid,
        store=store_session,
        match_how="prefix",
    )
    list_metapartitions = list(generator)
    mp_list = list_metapartitions[0]


def test_align_datasets_exact(dataset, evaluation_dataset, store_session):
    with pytest.raises(RuntimeError):
        list(
            align_datasets(
                left_dataset_uuid=dataset.uuid,
                right_dataset_uuid=evaluation_dataset.uuid,
                store=store_session,
                match_how="exact",
            )
        )

    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=dataset.uuid,
        store=store_session,
        match_how="exact",
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    # Two separate cluster_groups (e.g. cluster_1*)
    assert len(list_metapartitions) == 2

    mp_list = list_metapartitions[0]
    assert len(mp_list) == 2, [mp.label for mp in mp_list]
    assert [mp.label for mp in mp_list] == ["cluster_1", "cluster_1"]

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 2, [mp.label for mp in mp_list]
    assert [mp.label for mp in mp_list] == ["cluster_2", "cluster_2"]


def test_align_datasets_left(dataset, evaluation_dataset, store_session):
    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=evaluation_dataset.uuid,
        store=store_session,
        match_how="left",
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    assert len(list_metapartitions) == len(dataset.partitions)

    mp_list = list_metapartitions[0]
    assert len(mp_list) == 5, [mp.label for mp in mp_list]
    expected = ["cluster_1", "cluster_1_1", "cluster_1_2", "cluster_2_1", "cluster_2_2"]
    assert [mp.label for mp in mp_list] == expected

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 5, [mp.label for mp in mp_list]
    expected = ["cluster_2", "cluster_1_1", "cluster_1_2", "cluster_2_1", "cluster_2_2"]
    assert [mp.label for mp in mp_list] == expected


def test_align_datasets_right(dataset, evaluation_dataset, store_session):
    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=evaluation_dataset.uuid,
        store=store_session,
        match_how="right",
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    assert len(list_metapartitions) == len(evaluation_dataset.partitions)

    mp_list = list_metapartitions[0]
    assert len(mp_list) == 3, [mp.label for mp in mp_list]
    expected = ["cluster_1_1", "cluster_1", "cluster_2"]
    assert [mp.label for mp in mp_list] == expected

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 3, [mp.label for mp in mp_list]
    expected = ["cluster_1_2", "cluster_1", "cluster_2"]
    assert [mp.label for mp in mp_list] == expected

    mp_list = list_metapartitions[2]
    assert len(mp_list) == 3, [mp.label for mp in mp_list]
    expected = ["cluster_2_1", "cluster_1", "cluster_2"]
    assert [mp.label for mp in mp_list] == expected

    mp_list = list_metapartitions[3]
    assert len(mp_list) == 3, [mp.label for mp in mp_list]
    expected = ["cluster_2_2", "cluster_1", "cluster_2"]
    assert [mp.label for mp in mp_list] == expected


def test_align_datasets_callable(dataset, evaluation_dataset, store_session):
    def comp(left, right):
        return left == right

    with pytest.raises(RuntimeError):
        list(
            align_datasets(
                left_dataset_uuid=dataset.uuid,
                right_dataset_uuid=evaluation_dataset.uuid,
                store=store_session,
                match_how=comp,
            )
        )

    generator = align_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=dataset.uuid,
        store=store_session,
        match_how=comp,
    )
    assert isinstance(generator, types.GeneratorType)
    list_metapartitions = list(generator)

    # Two separate cluster_groups (e.g. cluster_1*)
    assert len(list_metapartitions) == 2

    mp_list = list_metapartitions[0]
    assert len(mp_list) == 2, [mp.label for mp in mp_list]
    assert [mp.label for mp in mp_list] == ["cluster_1", "cluster_1"]

    mp_list = list_metapartitions[1]
    assert len(mp_list) == 2, [mp.label for mp in mp_list]
    assert [mp.label for mp in mp_list] == ["cluster_2", "cluster_2"]


def test_merge_metapartitions():
    df = pd.DataFrame({"P": [1, 1], "L": [1, 2], "TARGET": [1, 2]})
    df_2 = pd.DataFrame({"P": [1], "info": "a"})
    mp = MetaPartition(label="cluster_1", data={"core": df, "helper": df_2})
    df_3 = pd.DataFrame({"P": [1, 1], "L": [1, 2], "PRED": [0.1, 0.2]})

    mp2 = MetaPartition(label="cluster_1", data={"predictions": df_3})
    merged_mp = MetaPartition.merge_metapartitions(metapartitions=[mp, mp2])

    df = pd.DataFrame(
        {
            "P": [1, 1],
            "L": [1, 2],
            "TARGET": [1, 2],
            "info": ["a", "a"],
            "PRED": [0.1, 0.2],
        }
    )

    assert merged_mp.label == "cluster_1"
    assert len(merged_mp.data) == 3
