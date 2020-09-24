import pickle

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.factory import DatasetFactory
from kartothek.io.dask._shuffle import _KTK_HASH_BUCKET, _hash_bucket
from kartothek.io.dask.dataframe import store_dataset_from_ddf, update_dataset_from_ddf
from kartothek.io.iter import read_dataset_as_dataframes__iterator


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
@pytest.mark.parametrize("func", [update_dataset_from_ddf, store_dataset_from_ddf])
def test_update_shuffle_buckets(
    store_factory,
    unique_primaries,
    unique_secondaries,
    num_buckets,
    repartition,
    npartitions,
    bucket_by,
    func,
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

    dataset_comp = func(
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
