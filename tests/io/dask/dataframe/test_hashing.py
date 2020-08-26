import pandas as pd
import pandas.testing as pdt

from kartothek.io.dask.dataframe import hash_dataset


def test_hash_dataset(dataset_with_index_factory):
    hh = (
        hash_dataset(factory=dataset_with_index_factory)
        .compute()
        .reset_index(drop=True)
    )

    expected = pd.Series([11462879952839863487, 12568779102514529673], dtype="uint64")
    assert len(hh) == len(dataset_with_index_factory.partitions)
    pdt.assert_series_equal(hh, expected)


def test_hash_dataset_subset(dataset_with_index_factory):
    hh = (
        hash_dataset(factory=dataset_with_index_factory, subset=["TARGET"])
        .compute()
        .reset_index(drop=True)
    )

    expected = pd.Series([11358988112447789330, 826468140851422801], dtype="uint64")
    assert len(hh) == len(dataset_with_index_factory.partitions)
    pdt.assert_series_equal(hh, expected)


def test_hash_dataset_group_keys(dataset_with_index_factory):

    group_keys = ["L"]
    hh = hash_dataset(
        factory=dataset_with_index_factory, group_key=group_keys
    ).compute()

    expected = pd.Series(
        [11462879952839863487, 12568779102514529673],
        dtype="uint64",
        index=pd.Index([1, 2], name="L"),
    )
    pdt.assert_series_equal(hh, expected)


def test_hash_dataset_group_keys_subset(dataset_with_index_factory):

    group_keys = ["P"]
    hh = hash_dataset(
        factory=dataset_with_index_factory, group_key=group_keys, subset=["TARGET"]
    ).compute()

    expected = pd.Series(
        [11358988112447789330, 826468140851422801],
        index=pd.Index([1, 2], name="P"),
        dtype="uint64",
    )
    pdt.assert_series_equal(hh, expected)


def test_hash_dataset_group_keys_subset_subset_groupkey(dataset_with_index_factory):

    group_keys = ["P"]
    hh = hash_dataset(
        factory=dataset_with_index_factory, group_key=group_keys, subset=["P", "TARGET"]
    ).compute()

    expected = pd.Series(
        [7554402398462747209, 1687604933839263903],
        index=pd.Index([1, 2], name="P"),
        dtype="uint64",
    )
    pdt.assert_series_equal(hh, expected)
