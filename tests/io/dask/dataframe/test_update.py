# -*- coding: utf-8 -*-
# pylint: disable=E1101

import pickle

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from kartothek.io.dask.dataframe import update_dataset_from_ddf
from kartothek.io.iter import read_dataset_as_dataframes__iterator
from kartothek.io.testing.update import *  # noqa


@pytest.fixture
def bound_update_dataset():
    return _update_dataset


def _unwrap_partition(part):
    return next(iter(dict(part["data"]).values()))


def _update_dataset(partitions, *args, **kwargs):
    if any(partitions):
        table_name = next(iter(dict(partitions[0]["data"]).keys()))
        delayed_partitions = [
            dask.delayed(_unwrap_partition)(part) for part in partitions
        ]
        partitions = dd.from_delayed(delayed_partitions)
    else:
        table_name = "core"
        partitions = None
    ddf = update_dataset_from_ddf(partitions, *args, table=table_name, **kwargs)

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    return ddf.compute()


def _return_none():
    return None


@pytest.mark.parametrize("unique_primaries", [1, 4])
@pytest.mark.parametrize("unique_secondaries", [1, 3])
@pytest.mark.parametrize("num_buckets", [1, 5])
@pytest.mark.parametrize("repartition", [1, 2])
@pytest.mark.parametrize("npartitions", [5, 10])
def test_update_shuffle_buckets(
    store_factory,
    metadata_version,
    unique_primaries,
    unique_secondaries,
    num_buckets,
    repartition,
    npartitions,
):
    """
    Assert that certain properties are always given for the output dataset
    no matter how the input data distribution looks like

    Properties to assert:
    * All partitions have a unique value for its correspondent primary key
    * number of partitions is at least one per unique partition value, at
      most ``num_buckets`` per primary partition value.
    * If we demand a column to be sorted it is per partition monotonic
    """
    primaries = np.arange(unique_primaries)
    secondary = np.arange(unique_secondaries)
    num_rows = 100
    primaries = np.repeat(primaries, np.ceil(num_rows / unique_primaries))[:num_rows]
    secondary = np.repeat(secondary, np.ceil(num_rows / unique_secondaries))[:num_rows]
    # ensure that there is an unsorted column uncorrelated
    # to the primary and secondary columns which can be sorted later on per partition
    unsorted_column = np.arange(num_rows)
    np.random.shuffle(unsorted_column)
    np.random.shuffle(primaries)
    np.random.shuffle(secondary)

    df = pd.DataFrame(
        {"primary": primaries, "secondary": secondary, "sorted_column": unsorted_column}
    )

    # shuffle all rows. properties of result should be reproducible
    df = df.sample(frac=1).reset_index(drop=True)
    ddf = dd.from_pandas(df, npartitions=npartitions)

    dataset_comp = update_dataset_from_ddf(
        ddf,
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="core",
        secondary_indices="secondary",
        shuffle=True,
        repartition_ratio=repartition,
        num_buckets=num_buckets,
        sort_partitions_by="sorted_column",
        default_metadata_version=metadata_version,
        partition_on=["primary"],
    )

    s = pickle.dumps(dataset_comp, pickle.HIGHEST_PROTOCOL)
    dataset_comp = pickle.loads(s)

    dataset = dataset_comp.compute()
    dataset = dataset.load_all_indices(store_factory())

    assert len(dataset.partitions) <= num_buckets * unique_primaries
    assert len(dataset.partitions) >= unique_primaries
    assert len(dataset.indices) == 2

    assert set(dataset.indices["primary"].index_dct.keys()) == set(
        range(unique_primaries)
    )
    assert (
        list(map(lambda x: len(x), dataset.indices["primary"].index_dct.values()))
        <= [num_buckets] * unique_primaries
    )

    assert set(dataset.indices["secondary"].index_dct.keys()) == set(
        range(unique_secondaries)
    )

    for data_dct in read_dataset_as_dataframes__iterator(
        dataset_uuid=dataset.uuid, store=store_factory
    ):
        df = data_dct["core"]
        assert len(df.primary.unique()) == 1
        assert df.sorted_column.is_monotonic

    # update the dataset
    # do not use partition_on since it should be interfered from the existing dataset
    tasks = update_dataset_from_ddf(
        ddf,
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="core",
        shuffle=True,
        repartition_ratio=repartition,
        num_buckets=num_buckets,
        sort_partitions_by="sorted_column",
        default_metadata_version=metadata_version,
    )

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    updated_dataset = tasks.compute()

    assert len(updated_dataset.partitions) == 2 * len(dataset.partitions)

    # Not allowed to use different partition_on
    with pytest.raises(
        ValueError, match="Incompatible set of partition keys encountered."
    ):
        update_dataset_from_ddf(
            ddf,
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=True,
            repartition_ratio=repartition,
            partition_on=["sorted_column"],
            num_buckets=num_buckets,
            sort_partitions_by="sorted_column",
            default_metadata_version=metadata_version,
        )

    # Not allowed to update with indices which do not yet exist in dataset
    with pytest.raises(ValueError, match="indices"):
        update_dataset_from_ddf(
            ddf,
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=True,
            partition_on=["primary"],
            repartition_ratio=repartition,
            secondary_indices=["sorted_column"],
            num_buckets=num_buckets,
            sort_partitions_by="sorted_column",
            default_metadata_version=metadata_version,
        )

    # Check that delayed objects are allowed as delete scope.
    tasks = update_dataset_from_ddf(
        None,
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="core",
        shuffle=True,
        repartition_ratio=repartition,
        num_buckets=num_buckets,
        sort_partitions_by="sorted_column",
        default_metadata_version=metadata_version,
        delete_scope=dask.delayed(_return_none)(),
    )

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    tasks.compute()
