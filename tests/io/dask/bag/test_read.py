import pickle
from functools import partial

import pandas as pd
import pytest

from kartothek.io.dask.bag import (
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
)
from kartothek.io.iter import store_dataframes_as_dataset__iter
from kartothek.io.testing.read import *  # noqa


@pytest.fixture(params=["dataframe", "metapartition"])
def output_type(request):
    return request.param


def _load_dataframes(output_type, *args, **kwargs):
    if output_type == "dataframe":
        func = read_dataset_as_dataframe_bag
    elif output_type == "metapartition":
        func = read_dataset_as_metapartitions_bag
    tasks = func(*args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    result = tasks.compute()
    return result


@pytest.fixture()
def bound_load_dataframes(output_type):
    return partial(_load_dataframes, output_type)


def test_read_dataset_as_dataframes_partition_size(store_factory, metadata_version):
    cluster1 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [1, 2], "Content": ["cluster1", "cluster1"]}
    )
    cluster2 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [2, 3], "Content": ["cluster2", "cluster2"]}
    )
    cluster3 = pd.DataFrame({"A": [1], "B": [20], "C": [1], "Content": ["cluster3"]})
    cluster4 = pd.DataFrame(
        {"A": [2, 2], "B": [10, 10], "C": [1, 2], "Content": ["cluster4", "cluster4"]}
    )
    clusters = [cluster1, cluster2, cluster3, cluster4]
    partitions = [{"data": [("data", c)]} for c in clusters]

    store_dataframes_as_dataset__iter(
        df_generator=partitions,
        store=store_factory,
        dataset_uuid="partitioned_uuid",
        metadata_version=metadata_version,
    )
    for func in [read_dataset_as_dataframe_bag, read_dataset_as_metapartitions_bag]:
        bag = func(
            dataset_uuid="partitioned_uuid", store=store_factory, partition_size=None
        )
        assert bag.npartitions == 4
        bag = func(
            dataset_uuid="partitioned_uuid", store=store_factory, partition_size=2
        )
        assert bag.npartitions == 2
