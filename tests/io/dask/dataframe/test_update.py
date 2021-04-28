import datetime

import dask
import dask.dataframe as dd
import pandas as pd
import pandas.testing as pdt
import pytest
from tests.io.common.conftest import update_dataset_dataframe

from kartothek.io.dask.dataframe import update_dataset_from_ddf
from kartothek.io.eager import read_table
from kartothek.io_components.metapartition import SINGLE_TABLE


@pytest.fixture
def bound_update_dataset():
    return update_dataset_dataframe


def _return_none():
    return None


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


@pytest.fixture(params=["other_table", SINGLE_TABLE])
def alternative_table_name(request):
    return request.param


def test_update_different_table_name(
    meta_partitions_dataframe_alternative_table_name,
    bound_update_dataset,
    metadata_version,
    store,
    alternative_table_name,
):
    first_partition = meta_partitions_dataframe_alternative_table_name[0]
    dataset = update_dataset_from_ddf(
        dd.from_pandas(first_partition.data, npartitions=1),
        store,
        dataset_uuid="dataset_uuid",
        table=alternative_table_name,
        shuffle=True,
        partition_on=["P"],
    ).compute()
    assert dataset is not None
    second_partition = meta_partitions_dataframe_alternative_table_name[1]
    dataset_updated = update_dataset_from_ddf(
        dd.from_pandas(second_partition.data, npartitions=1),
        store,
        dataset_uuid="dataset_uuid",
        metadata={"extra": "metadata"},
        table=alternative_table_name,
        shuffle=True,
        partition_on=["P"],
    ).compute()
    assert dataset_updated is not None
    df = read_table(store=store, dataset_uuid="dataset_uuid",)
    assert df is not None
    expected_df = pd.DataFrame(
        {
            "P": [1, 2],
            "L": [1, 2],
            "TARGET": [1, 2],
            "DATE": [datetime.date(2010, 1, 1), datetime.date(2009, 12, 31)],
        }
    )

    pdt.assert_frame_equal(df, expected_df, check_dtype=True, check_like=True)
