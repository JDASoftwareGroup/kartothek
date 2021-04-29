import pickle
from functools import partial

import pandas as pd
import pytest
from pandas import testing as pdt

from kartothek.io.dask.bag import (
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
)
from kartothek.io.iter import store_dataframes_as_dataset__iter


@pytest.fixture(params=["dataframe"])
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
    partitions = [cluster1, cluster2, cluster3, cluster4]

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


@pytest.mark.parametrize("specify_name", [True, False])
def test_read_table_dask_bag(dataset_one_name, store_session, specify_name):
    """
    Tests reading a dataset with either default or alternative table name as
    Dask bag. Table name is either specified or mot specified.
    """
    if specify_name != "No":
        # Name specification not implemented for dask bag
        pytest.skip()
    else:
        ddf = read_dataset_as_dataframe_bag(
            dataset_uuid=dataset_one_name.uuid, store=store_session,
        )

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    result = ddf.compute()

    import datetime

    expected_df = pd.DataFrame(
        {
            "P": [1, 2],
            "L": [1, 2],
            "TARGET": [1, 2],
            "DATE": [datetime.date(2010, 1, 1), datetime.date(2009, 12, 31)],
        }
    )

    # No stability of partitions
    df_actual = pd.concat(result).sort_values(by="P").reset_index(drop=True)
    pdt.assert_frame_equal(df_actual, expected_df, check_dtype=True, check_like=True)


# We would expect this test to gracefully fail with a RuntimeError.
# Instead, a KeyError is thrown.
@pytest.mark.xfail
def test_read_table_dask_ddf_multitable(dataset_two_table_names, store_session):
    """
    Tests reading a dataset with two differing tables names as  Dask dataframe.
    No table name is specified while reading.
    """
    with pytest.raises(RuntimeError):
        _ = read_dataset_as_dataframe_bag(
            dataset_uuid=dataset_two_table_names.uuid, store=store_session,
        )
