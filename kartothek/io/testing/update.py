# -*- coding: utf-8 -*-
# pylint: disable=E1101


from functools import partial

import numpy as np
import pandas as pd
import pytest

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.naming import DEFAULT_METADATA_VERSION
from kartothek.core.testing import TIME_TO_FREEZE_ISO
from kartothek.io.eager import (
    read_dataset_as_metapartitions,
    store_dataframes_as_dataset,
)
from kartothek.io.iter import read_dataset_as_dataframes__iterator


def test_update_dataset_with_partitions__reducer(
    store_factory, metadata_version, bound_update_dataset, mocker, store
):
    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": [1]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={1: ["cluster_1"]})},
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": [2]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={2: ["cluster_2"]})},
        },
    ]
    dataset = bound_update_dataset(
        partitions,
        store=store_factory,
        metadata={"dataset": "metadata"},
        dataset_uuid="dataset_uuid",
        default_metadata_version=metadata_version,
        secondary_indices=["p"],
    )
    dataset = dataset.load_index("p", store)

    part3 = {
        "label": "cluster_3",
        "data": [("core", pd.DataFrame({"p": [3]}))],
        "indices": {"p": ExplicitSecondaryIndex("p", index_dct={3: ["cluster_3"]})},
    }

    dataset_updated = bound_update_dataset(
        [part3],
        store=store_factory,
        delete_scope=[{"p": 1}],
        metadata={"extra": "metadata"},
        dataset_uuid="dataset_uuid",
        default_metadata_version=metadata_version,
        secondary_indices=["p"],
    )
    dataset_updated = dataset_updated.load_index("p", store)

    # First check the index. if they are as expected, use them to determine label name

    exp_idx_values = {1, 2}
    ind = dataset.indices["p"]
    ind_updated = dataset_updated.indices["p"]
    assert set(ind.index_dct.keys()) == exp_idx_values
    exp_updated_idx_values = {2, 3}
    assert set(ind_updated.index_dct.keys()) == exp_updated_idx_values

    assert ind_updated.index_dct[2] == ind.index_dct[2]
    assert ind.index_dct[1] != ind_updated.index_dct[3]

    expected_metadata = {"dataset": "metadata", "extra": "metadata"}

    # We do not mock the time resolution of this test since otherwise we cannot ensure
    # that the indices are not overwritten
    expected_metadata["creation_time"] = dataset_updated.metadata["creation_time"]

    assert dataset_updated.metadata == expected_metadata
    assert dataset_updated.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # 1 dataset metadata file and 2 index file and 3 partition files

    # common metadata for v4 datasets (1 table)
    expected_number_files = 7

    assert len(store_files) == expected_number_files

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    stored_dataset = stored_dataset.load_index("p", store)
    assert dataset_updated == stored_dataset


def test_update_dataset_with_partitions_no_index_input_info(
    store_factory, metadata_version, bound_update_dataset, store
):
    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": [1]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={1: ["cluster_1"]})},
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": [2]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={2: ["cluster_2"]})},
        },
    ]
    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        metadata={"dataset": "metadata"},
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    # The input information doesn't explicitly provide index information
    # Since the dataset has an index, it must be updated either way
    part3 = {"label": "cluster_3", "data": [("core", pd.DataFrame({"p": [3]}))]}
    dataset_updated = bound_update_dataset(
        [part3],
        store=store_factory,
        dataset_uuid=dataset.uuid,
        delete_scope=[{"p": 1}],
        metadata={"extra": "metadata"},
        default_metadata_version=metadata_version,
        secondary_indices=["p"],
    )
    dataset_updated = dataset_updated.load_all_indices(store)
    assert 3 in dataset_updated.indices["p"].to_dict()


def test_update_dataset_with_partitions__reducer_delete_only(
    store_factory, metadata_version, frozen_time_em, bound_update_dataset, store
):
    partitions = [
        {
            "label": "cluster_1",
            "data": [("core", pd.DataFrame({"p": [1]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={1: ["cluster_1"]})},
        },
        {
            "label": "cluster_2",
            "data": [("core", pd.DataFrame({"p": [2]}))],
            "indices": {"p": ExplicitSecondaryIndex("p", index_dct={2: ["cluster_2"]})},
        },
    ]
    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        metadata={"dataset": "metadata"},
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )
    dataset = dataset.load_index("p", store)

    empty_part = []
    dataset_updated = bound_update_dataset(
        [empty_part],
        store=store_factory,
        dataset_uuid="dataset_uuid",
        delete_scope=[{"p": 1}],
        metadata={"extra": "metadata"},
        default_metadata_version=metadata_version,
        secondary_indices=["p"],
    )
    dataset_updated = dataset_updated.load_index("p", store)

    assert sorted(dataset.partitions) == ["cluster_1", "cluster_2"]
    assert list(dataset_updated.partitions) == ["cluster_2"]

    store_files = list(store.keys())
    # 1 dataset metadata file and 1 index file and 2 partition files
    # note: the update writes a new index file but due to frozen_time this gets
    # the same name as the previous one and overwrites it.
    expected_number_files = 4
    # common metadata for v4 datasets (1 table)
    expected_number_files += 1
    assert len(store_files) == expected_number_files

    assert dataset.indices["p"].index_dct == {1: ["cluster_1"], 2: ["cluster_2"]}
    assert dataset_updated.indices["p"].index_dct == {2: ["cluster_2"]}

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    stored_dataset = stored_dataset.load_index("p", store)
    assert dataset_updated == stored_dataset


def test_update_dataset_with_partitions__reducer_partitions(
    store_factory, frozen_time_em, bound_update_dataset
):

    assert set(store_factory().keys()) == set()

    df1 = pd.DataFrame(
        {"P": [1, 2, 3, 1, 2, 3], "L": [1, 1, 1, 1, 1, 1], "TARGET": np.arange(10, 16)}
    )
    df2 = df1.copy(deep=True)
    df2.L = 2
    df2.TARGET += 2
    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df1)],
            "indices": {"L": {k: ["cluster_1"] for k in df1["L"].unique()}},
        },
        {
            "label": "cluster_2",
            "data": [("core", df2)],
            "indices": {"L": {k: ["cluster_2"] for k in df2["L"].unique()}},
        },
    ]
    dataset = store_dataframes_as_dataset(
        dfs=df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        partition_on=["P"],
        metadata_version=4,
    )
    dataset_loadedidx = dataset.load_all_indices(store=store_factory())
    cluster_1_label = (
        dataset_loadedidx.indices["L"].eval_operator(op="==", value=1).pop()
    )
    cluster_1_label = cluster_1_label.split("/")[-1]
    cluster_2_label = (
        dataset_loadedidx.indices["L"].eval_operator(op="==", value=2).pop()
    )
    cluster_2_label = cluster_2_label.split("/")[-1]

    df3 = df2.copy(deep=True)
    df3.TARGET -= 5

    part3 = {
        "label": "cluster_3",
        "data": {"core": df3},
        "indices": {"L": {k: ["cluster_3"] for k in df3["L"].unique()}},
    }

    dataset_updated = bound_update_dataset(
        [part3],
        store=store_factory,
        dataset_uuid="dataset_uuid",
        delete_scope=[{"L": 2}],
        metadata={"extra": "metadata"},
        partition_on=["P"],
        secondary_indices=["L"],
    )
    dataset_updated_loadedidx = dataset_updated.load_all_indices(store=store_factory())
    cluster_3_labels = dataset_updated_loadedidx.indices["L"].eval_operator(
        op="==", value=2
    )

    cluster_3_label = {c3_label.split("/")[-1] for c3_label in cluster_3_labels}
    assert len(cluster_3_label) == 1
    cluster_3_label = cluster_3_label.pop()
    exp_partitions = [
        "P=1/{}".format(cluster_1_label),
        "P=1/{}".format(cluster_3_label),
        "P=2/{}".format(cluster_1_label),
        "P=2/{}".format(cluster_3_label),
        "P=3/{}".format(cluster_1_label),
        "P=3/{}".format(cluster_3_label),
    ]
    assert sorted(exp_partitions) == sorted(dataset_updated.partitions.keys())
    updated_idx_keys = sorted(dataset_updated.indices.keys())
    assert sorted(dataset.indices.keys()) == updated_idx_keys

    expected_new_idx = {}
    for k, v in dataset_loadedidx.indices["P"].index_dct.items():
        val = [pl.replace(cluster_2_label, cluster_3_label) for pl in v]
        expected_new_idx[k] = val

    updated_P_idx_dct = dataset_updated_loadedidx.indices["P"].index_dct

    assert sorted(expected_new_idx.keys()) == sorted(updated_P_idx_dct.keys())

    for k, v in updated_P_idx_dct.items():
        assert sorted(expected_new_idx[k]) == sorted(v)


def test_update_dataset_with_partitions__reducer_nonexistent(
    store_factory, metadata_version, frozen_time_em, bound_update_dataset, store
):

    part3 = {
        "label": "cluster_3",
        "data": [("core", pd.DataFrame({"p": [3]}))],
        "indices": {"p": ExplicitSecondaryIndex("p", index_dct={3: ["cluster_3"]})},
    }
    dataset_updated = bound_update_dataset(
        [part3],
        store=store_factory,
        dataset_uuid="dataset_uuid",
        delete_scope=[{"p": 1}],
        metadata={"extra": "metadata"},
        default_metadata_version=metadata_version,
        secondary_indices=["p"],
    )
    dataset_updated = dataset_updated.load_index("p", store)
    ind_updated = dataset_updated.indices["p"]
    cluster_3_label = ind_updated.eval_operator(op="==", value=3).pop()

    expected_metadata = {"extra": "metadata"}

    expected_metadata["creation_time"] = TIME_TO_FREEZE_ISO

    assert dataset_updated.metadata == expected_metadata
    assert list(dataset_updated.partitions) == [cluster_3_label]

    updated_part_c3 = dataset_updated.partitions[cluster_3_label]

    assert updated_part_c3.label == cluster_3_label
    assert dataset_updated.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # 1 dataset metadata file and 1 index file and 1 partition files
    # note: the update writes a new index file but due to frozen_time this gets
    # the same name as the previous one and overwrites it.
    expected_number_files = 3

    # common metadata for v4 datasets (1 table)
    expected_number_files += 1
    assert len(store_files) == expected_number_files
    exp_updated_idx = {3: [cluster_3_label]}
    assert dataset_updated.indices["p"].index_dct == exp_updated_idx

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    stored_dataset = stored_dataset.load_index("p", store)
    assert dataset_updated == stored_dataset


@pytest.mark.parametrize(
    "dfs,ok",
    [
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int32),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int16),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int16),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int32),
                        "X": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.uint64),
                    }
                ),
            ],
            False,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int64),
                        "Y": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            False,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1, 2], dtype=np.int64),
                        "X": pd.Series([1, 2], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([3], dtype=np.int64),
                        "X": pd.Series([3], dtype=np.uint64),
                    }
                ),
            ],
            False,
        ),
    ],
)
def test_schema_check_update(dfs, ok, store_factory, bound_update_dataset):
    df_list = [{"label": "cluster_1", "data": [("core", df)]} for df in dfs]
    store_dataframes_as_dataset(
        dfs=df_list[:1],
        store=store_factory,
        dataset_uuid="dataset_uuid",
        partition_on=["P"],
        metadata_version=4,
    )
    pipe = partial(
        bound_update_dataset,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        partition_on=["P"],
    )

    if ok:
        pipe(df_list[1:])
    else:
        with pytest.raises(
            Exception,
            match=r"Schemas\sfor\stable\s\\*'core\\*'\sof\sdataset\s\\*'dataset_uuid\\*'\sare\snot\scompatible!",
        ):
            pipe(df_list[1:])


def test_sort_partitions_by(
    store_factory, metadata_version, frozen_time_em, bound_update_dataset
):
    df1 = pd.DataFrame({"P": [3], "L": [1], "TARGET": [1]})
    df2 = pd.DataFrame(
        {
            "P": [1, 2, 3, 1, 2, 3],
            "L": [1, 1, 1, 1, 1, 1],
            "TARGET": list(reversed(np.arange(10, 16))),
        }
    )

    df3 = pd.DataFrame(
        {
            "P": [1, 2, 3, 1, 2, 3],
            "L": [1, 1, 1, 1, 1, 1],
            "TARGET": [88, 1, 5, 99, 12, 11],
        }
    )

    df_list = [{"label": "cluster_1", "data": [("core", df1)]}]
    new_partitions = [
        {"label": "cluster_2", "data": [("core", df2)]},
        {"label": "cluster_3", "data": [("core", df3)]},
    ]

    store_dataframes_as_dataset(
        dfs=df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    bound_update_dataset(
        new_partitions,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata={"extra": "metadata"},
        default_metadata_version=metadata_version,
        sort_partitions_by=["TARGET"],
    )

    # Check that the `sort_partitions_by` column is indeed sorted monotonically among partitions
    for label_df_tupl in read_dataset_as_dataframes__iterator(
        store=store_factory, dataset_uuid="dataset_uuid"
    ):
        for _, df in label_df_tupl.items():
            assert (df.TARGET == sorted(df.TARGET)).all()


def test_metadata_version(
    store_factory,
    bound_update_dataset,
    mock_default_metadata_version,
    backend_identifier,
):
    if backend_identifier in ("dask.dataframe", "dask.delayed"):
        pytest.skip()  # TODO: fix `io.dask.*.test_update._update_dataset`

    dataset_uuid = "dataset_uuid"
    partitions = [
        {"label": "cluster_1", "data": [("core", pd.DataFrame({"p": [1, 2]}))]},
        {"label": "cluster_2", "data": [("core", pd.DataFrame({"p": [2, 3]}))]},
    ]

    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=DEFAULT_METADATA_VERSION,
    )

    with pytest.raises(AssertionError, match="Traversed through mock"):
        # Try to commit data to dataset using a different metadata version
        # and different data format (format is mocked)
        # This does not raise when the `parse_input_to_metapartition`
        # argument is `default_metadata_version` instead of `metadata_version`
        new_partitions = ("core", pd.DataFrame({"p": [2, 3]}))
        bound_update_dataset(
            new_partitions,
            store=store_factory,
            dataset_uuid=dataset_uuid,
            default_metadata_version=mock_default_metadata_version,
        )

    mps = read_dataset_as_metapartitions(store=store_factory, dataset_uuid=dataset_uuid)
    assert len(mps) == len(dataset.partitions)


def test_raises_on_invalid_input(store_factory, bound_update_dataset):
    dataset_uuid = "dataset_uuid"
    partitions = [
        {"label": "cluster_1", "data": [("core", pd.DataFrame({"p": [1, 2]}))]},
        {"label": "cluster_2", "data": [("core", pd.DataFrame({"p": [2, 3]}))]},
    ]

    dataset = store_dataframes_as_dataset(
        dfs=partitions, store=store_factory, dataset_uuid=dataset_uuid
    )

    with pytest.raises(Exception):
        new_partitions = [({"stuff"}, [("something", {1, 2, 3})])]  # invalid format
        bound_update_dataset(
            new_partitions, store=store_factory, dataset_uuid=dataset_uuid
        )

    # Check no new partitions have been written to storage
    mps = read_dataset_as_metapartitions(store=store_factory, dataset_uuid=dataset_uuid)
    assert len(mps) == len(dataset.partitions)


@pytest.mark.parametrize("define_indices_on_partition", (False, True))
def test_raises_on_new_index_creation(
    backend_identifier, store_factory, bound_update_dataset, define_indices_on_partition
):
    if backend_identifier == "dask.dataframe" and define_indices_on_partition:
        pytest.skip()  # Constructs a dataframe which ignores index information passed as dict

    dataset_uuid = "dataset_uuid"
    index_column = "p"
    partitions = [
        {"label": "cluster_1", "data": [("core", pd.DataFrame({index_column: [1, 2]}))]}
    ]

    new_partition = {
        "label": "cluster_2",
        "data": [("core", pd.DataFrame({index_column: [2, 3]}))],
    }

    dataset_update_secondary_indices = [index_column]
    if define_indices_on_partition:
        dataset_update_secondary_indices = None
        new_partition["indices"] = {
            index_column: ExplicitSecondaryIndex(
                index_column,
                {
                    k: [new_partition["label"]]
                    for k in new_partition["data"][0][1][index_column].unique()
                },
            )
        }

    # Create dataset without secondary indices
    store_dataframes_as_dataset(
        dfs=partitions, store=store_factory, dataset_uuid=dataset_uuid
    )

    with pytest.raises(Exception, match="Incorrect indices provided for dataset"):
        bound_update_dataset(
            [new_partition],
            store=store_factory,
            dataset_uuid=dataset_uuid,
            secondary_indices=dataset_update_secondary_indices,
        )


@pytest.mark.parametrize("define_indices_on_partition", (False, True))
def test_update_first_time_with_secondary_indices(
    store_factory, bound_update_dataset, define_indices_on_partition
):
    """
    Check it is possible to create a new dataset with indices defined either on partition or using the
    `secondary_indices` kwarg. The intention of this test is to verify that there are no exceptions raised
    related to index validation when a dataset is created using an `update` function
    """
    dataset_uuid = "dataset_uuid"
    index_column = "p"
    new_partition = {
        "label": "cluster_1",
        "data": [("core", pd.DataFrame({index_column: [1, 2]}))],
    }

    dataset_update_secondary_indices = [index_column]
    if define_indices_on_partition:
        dataset_update_secondary_indices = None
        new_partition["indices"] = {
            index_column: ExplicitSecondaryIndex(
                index_column,
                {
                    k: [new_partition["label"]]
                    for k in new_partition["data"][0][1][index_column].unique()
                },
            )
        }
    bound_update_dataset(
        [new_partition],
        store=store_factory,
        dataset_uuid=dataset_uuid,
        secondary_indices=dataset_update_secondary_indices,
    )


def test_partition_on_null(store_factory, bound_update_dataset):  # gh-262
    keys = ["a", "b", "c", np.nan]
    values = range(len(keys))
    d = dict(zip(keys, values))
    df = (
        pd.DataFrame.from_dict(d, orient="index")
        .reset_index()
        .rename(columns={"index": "part", 0: "value"})
    )

    with pytest.raises(
        Exception, match=r"Original dataframe size .* on a column with null values."
    ):
        bound_update_dataset(
            [{"data": {"table": df}}],
            store=store_factory,
            dataset_uuid="a_unique_dataset_identifier",
            partition_on=["part"],
        )


def test_update_infers_partition_on(store_factory, bound_update_dataset, df_not_nested):
    dataset_uuid = "dataset_uuid"

    dataset = bound_update_dataset(
        [{"data": {"table": df_not_nested}}],
        dataset_uuid=dataset_uuid,
        store=store_factory,
        partition_on=df_not_nested.columns[0],
    )
    # update the dataset
    # do not use partition_on since it should be interfered from the existing dataset

    updated_dataset = bound_update_dataset(
        [{"data": {"table": df_not_nested}}],
        dataset_uuid=dataset_uuid,
        store=store_factory,
    )

    assert len(updated_dataset.partitions) == 2 * len(dataset.partitions)


def test_update_raises_incompatible_partition_keys(
    store_factory, bound_update_dataset, df_not_nested
):
    dataset_uuid = "dataset_uuid"
    bound_update_dataset(
        [{"data": {"table": df_not_nested}}],
        dataset_uuid=dataset_uuid,
        store=store_factory,
        partition_on=df_not_nested.columns[0],
    )
    # Not allowed to use different partition_on
    with pytest.raises(
        ValueError, match="Incompatible set of partition keys encountered."
    ):
        bound_update_dataset(
            [{"data": {"table": df_not_nested}}],
            dataset_uuid=dataset_uuid,
            store=store_factory,
            partition_on=df_not_nested.columns[1],
        )


def test_update_raises_incompatible_inidces(
    store_factory, bound_update_dataset, df_not_nested
):
    dataset_uuid = "dataset_uuid"
    bound_update_dataset(
        [{"data": {"table": df_not_nested}}],
        dataset_uuid=dataset_uuid,
        store=store_factory,
        secondary_indices=df_not_nested.columns[0],
    )
    # Not allowed to update with indices which do not yet exist in dataset
    with pytest.raises(ValueError, match="indices"):
        bound_update_dataset(
            [{"data": {"table": df_not_nested}}],
            dataset_uuid=dataset_uuid,
            store=store_factory,
            secondary_indices=df_not_nested.columns[1],
        )
