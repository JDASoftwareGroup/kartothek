# -*- coding: utf-8 -*-
# pylint: disable=E1101

import pickle

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.factory import DatasetFactory
from kartothek.io.dask._update import (
    _KTK_HASH_BUCKET,
    _hash_bucket,
    pack_payload,
    pack_payload_pandas,
    unpack_payload,
    unpack_payload_pandas,
)
from kartothek.io.dask.dataframe import update_dataset_from_ddf
from kartothek.io.iter import read_dataset_as_dataframes__iterator
from kartothek.io.testing.update import *  # noqa


@pytest.mark.parametrize("col", ["range", "range_duplicated", "random"])
def test_hash_bucket(col, num_buckets=5):
    df = pd.DataFrame(
        {
            "range": np.arange(10),
            "range_duplicated": np.repeat(np.arange(2), 5),
            "random": np.random.randint(0, 100, 10),
        }
    )
    hashed = _hash_bucket(df, [col], num_buckets)
    assert (hashed.groupby(col).agg({_KTK_HASH_BUCKET: "nunique"}) == 1).all().all()

    # Check that hashing is consistent for small dataframe sizes (where df.col.nunique() < num_buckets)
    df_sample = df.iloc[[0, 7]]
    hashed_sample = _hash_bucket(df_sample, [col], num_buckets)
    expected = hashed.loc[df_sample.index]
    pdt.assert_frame_equal(expected, hashed_sample)


def test_hashing_determinism():
    """Make sure that the hashing algorithm used by pandas is independent of any context variables"""
    df = pd.DataFrame({"range": np.arange(10)})
    hashed = _hash_bucket(df, ["range"], 5)
    expected = pd.DataFrame(
        {
            "range": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            _KTK_HASH_BUCKET: np.uint8([0, 0, 1, 2, 0, 3, 2, 0, 1, 4]),
        }
    )
    pdt.assert_frame_equal(hashed, expected)


@pytest.fixture
def bound_update_dataset():
    return _update_dataset


def _unwrap_partition(part):
    return next(iter(dict(part["data"]).values()))


def _update_dataset(partitions, *args, **kwargs):
    # TODO: fix the parsing below to adapt for all supported formats (see: parse_input_to_metapartition)
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


@pytest.mark.parametrize("bucket_by", [None, "range"])
def test_update_shuffle_no_partition_on(store_factory, bucket_by):
    df = pd.DataFrame(
        {
            "range": np.arange(10),
            "range_duplicated": np.repeat(np.arange(2), 5),
            "random": np.random.randint(0, 100, 10),
        }
    )
    ddf = dd.from_pandas(df, npartitions=10)

    with pytest.raises(
        ValueError, match="``num_buckets`` must not be None when shuffling data."
    ):
        update_dataset_from_ddf(
            ddf,
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="table",
            shuffle=True,
            num_buckets=None,
            bucket_by=bucket_by,
        ).compute()

    res_default = update_dataset_from_ddf(
        ddf,
        store_factory,
        dataset_uuid="output_dataset_uuid_default",
        table="table",
        shuffle=True,
        bucket_by=bucket_by,
    ).compute()
    assert len(res_default.partitions) == 1

    res = update_dataset_from_ddf(
        ddf,
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="table",
        shuffle=True,
        num_buckets=2,
        bucket_by=bucket_by,
    ).compute()

    assert len(res.partitions) == 2


@pytest.mark.parametrize("unique_primaries", [1, 4])
@pytest.mark.parametrize("unique_secondaries", [1, 3])
@pytest.mark.parametrize("num_buckets", [1, 5])
@pytest.mark.parametrize("repartition", [1, 2])
@pytest.mark.parametrize("npartitions", [5, 10])
@pytest.mark.parametrize("bucket_by", [None, "sorted_column"])
def test_update_shuffle_buckets(
    store_factory,
    metadata_version,
    unique_primaries,
    unique_secondaries,
    num_buckets,
    repartition,
    npartitions,
    bucket_by,
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
    unsorted_column = np.repeat(np.arange(100 / 10), 10)
    np.random.shuffle(unsorted_column)
    np.random.shuffle(primaries)
    np.random.shuffle(secondary)

    df = pd.DataFrame(
        {"primary": primaries, "secondary": secondary, "sorted_column": unsorted_column}
    )
    secondary_indices = ["secondary"]
    expected_num_indices = 2  # One primary

    # used for tests later on to
    if bucket_by:
        secondary_indices.append(bucket_by)
        expected_num_indices = 3

    # shuffle all rows. properties of result should be reproducible
    df = df.sample(frac=1).reset_index(drop=True)
    ddf = dd.from_pandas(df, npartitions=npartitions)

    dataset_comp = update_dataset_from_ddf(
        ddf,
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="core",
        secondary_indices=secondary_indices,
        shuffle=True,
        bucket_by=bucket_by,
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

    assert len(dataset.indices) == expected_num_indices

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

    assert set(dataset.table_meta["core"].names) == {
        "primary",
        "secondary",
        "sorted_column",
    }

    factory = DatasetFactory("output_dataset_uuid", store_factory)
    factory.load_all_indices()

    if bucket_by:
        ind_df = factory.get_indices_as_dataframe(["primary", bucket_by])

        assert not ind_df.duplicated().any()

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
        bucket_by=bucket_by,
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
        bucket_by=bucket_by,
    )

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    tasks.compute()


@pytest.mark.parametrize("shuffle", [True, False])
def test_update_dataset_from_ddf_empty(store_factory, shuffle):
    with pytest.raises(ValueError, match="Cannot store empty datasets"):
        update_dataset_from_ddf(
            dask.dataframe.from_delayed([], meta=(("a", int),)),
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=shuffle,
            partition_on=["a"],
        ).compute()


def test_pack_payload(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df = dd.from_pandas(
        pd.concat([df_all_types] * 10, ignore_index=True), npartitions=3
    )
    size_before = df.memory_usage(deep=True).sum()

    packed_df = pack_payload(df, group_key=list(df.columns[-2:]))

    size_after = packed_df.memory_usage(deep=True).sum()

    assert (size_after < size_before).compute()


def test_pack_payload_empty(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df_empty = dd.from_pandas(df_all_types.iloc[:0], npartitions=1)

    group_key = [df_all_types.columns[-1]]
    pdt.assert_frame_equal(
        df_empty.compute(),
        unpack_payload(
            pack_payload(df_empty, group_key=group_key), unpack_meta=df_empty._meta
        ).compute(),
    )


def test_pack_payload_pandas(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df = pd.concat([df_all_types] * 10, ignore_index=True)
    size_before = df.memory_usage(deep=True).sum()

    packed_df = pack_payload_pandas(df, group_key=list(df.columns[-2:]))

    size_after = packed_df.memory_usage(deep=True).sum()

    assert size_after < size_before


def test_pack_payload_pandas_empty(df_all_types):
    # For a single row dataframe the packing actually has a few more bytes
    df_empty = df_all_types.iloc[:0]

    group_key = [df_all_types.columns[-1]]
    pdt.assert_frame_equal(
        df_empty,
        unpack_payload_pandas(
            pack_payload_pandas(df_empty, group_key=group_key), unpack_meta=df_empty
        ),
    )


@pytest.mark.parametrize("num_group_cols", [1, 4])
def test_pack_payload_roundtrip(df_all_types, num_group_cols):
    group_key = list(df_all_types.columns[-num_group_cols:])
    df_all_types = dd.from_pandas(df_all_types, npartitions=2)
    pdt.assert_frame_equal(
        df_all_types.compute(),
        unpack_payload(
            pack_payload(df_all_types, group_key=group_key),
            unpack_meta=df_all_types._meta,
        ).compute(),
    )


@pytest.mark.parametrize("num_group_cols", [1, 4])
def test_pack_payload_pandas_roundtrip(df_all_types, num_group_cols):
    group_key = list(df_all_types.columns[-num_group_cols:])
    pdt.assert_frame_equal(
        df_all_types,
        unpack_payload_pandas(
            pack_payload_pandas(df_all_types, group_key=group_key),
            unpack_meta=df_all_types,
        ),
    )
