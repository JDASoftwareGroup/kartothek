import pandas as pd
import pytest

from kartothek.core.factory import DatasetFactory  # noqa: E402
from kartothek.io.dask.dataframe import read_dataset_as_ddf, repartition_ddf
from kartothek.io.iter import store_dataframes_as_dataset__iter


@pytest.mark.parametrize("frac", (1.0, 0.5))
@pytest.mark.parametrize(
    "df,target_nrows_per_partition",
    (
        (
            pd.DataFrame({"A": range(100), "part": ["A", "B", "C", "D"] * 25}),
            100,
        ),  # rep_ratio = 4
        (
            pd.DataFrame({"A": range(2 ** 7), "part": ["A", "B", "C", "D"] * 2 ** 5}),
            8,
        ),  # rep_ratio = 0.25
    ),
)
def test_repartition_ddf_exact_match(
    df, target_nrows_per_partition, frac, store_factory
):
    """
    Because we carefully choose the number of original partitions and the value of
    `target_nrows_per_partition`, we get exact matches for `target_nrows_per_partition`.
    This is not expected in a real-world scenario where the conditions are very unlikely.
    """
    dfs = [df]
    store_dataframes_as_dataset__iter(
        dfs, store=store_factory, dataset_uuid="test", partition_on=["part"],
    )

    ddf = read_dataset_as_ddf(store=store_factory, dataset_uuid="test")
    ddf = repartition_ddf(
        ddf,
        dataset_factory=DatasetFactory(
            dataset_uuid="test", store_factory=store_factory
        ),
        target_nrows_per_partition=target_nrows_per_partition,
        dataset_metadata_frac=frac,
    )

    assert (ddf.map_partitions(len).compute() == target_nrows_per_partition).all()


@pytest.mark.parametrize(
    "df,target_nrows_per_partition",
    (
        (
            pd.DataFrame({"A": range(100), "part": ["A", "B", "C", "D"] * 25}),
            97,
        ),  # rep_ratio = 3.88
        (pd.DataFrame({"A": range(10), "part": range(10)}), 1),  # rep_ratio = 1.0
    ),
)
def test_repartition_ddf_approximate_match(
    df, target_nrows_per_partition, store_factory
):
    dfs = [df]
    store_dataframes_as_dataset__iter(
        dfs, store=store_factory, dataset_uuid="test", partition_on=["part"],
    )

    ddf = read_dataset_as_ddf(store=store_factory, dataset_uuid="test")
    ddf = repartition_ddf(
        ddf,
        dataset_factory=DatasetFactory(
            dataset_uuid="test", store_factory=store_factory
        ),
        target_nrows_per_partition=target_nrows_per_partition,
        dataset_metadata_frac=0.3,
    )
    # this is somewhat arbitrary as original partition size may be non-constant
    margin_of_error = 0.5  # proportion
    partition_lengths = ddf.map_partitions(len).compute()
    assert (
        ((1 - margin_of_error) * target_nrows_per_partition <= partition_lengths)
        & (partition_lengths <= (1 + margin_of_error) * target_nrows_per_partition)
    ).all()
