import pickle

import dask
import dask.dataframe as dd
import pandas as pd
import pytest

from kartothek.io.dask.dataframe import update_dataset_from_ddf
from kartothek.io.testing.update import *  # noqa


@pytest.fixture
def bound_update_dataset():
    return _update_dataset


def _unwrap_partition(part):
    return next(iter(dict(part["data"]).values()))


def _update_dataset(partitions, *args, **kwargs):
    # TODO: Simplify once parse_input_to_metapartition is removed / obsolete
    if isinstance(partitions, pd.DataFrame):
        table_name = "core"
        partitions = dd.from_pandas(partitions, npartitions=1)
    elif any(partitions):
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


def test_delayed_as_delete_scope(store_factory, df_all_types):
    # Check that delayed objects are allowed as delete scope.
    tasks = update_dataset_from_ddf(
        dd.from_pandas(df_all_types, npartitions=1),
        store_factory,
        dataset_uuid="output_dataset_uuid",
        table="core",
        delete_scope=dask.delayed(_return_none)(),
    )

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    tasks.compute()


@pytest.mark.parametrize("shuffle", [True, False])
def test_update_dataset_from_ddf_empty(store_factory, shuffle):
    with pytest.raises(ValueError) as exc_info:
        update_dataset_from_ddf(
            dask.dataframe.from_delayed([], meta=(("a", int),)),
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=shuffle,
            partition_on=["a"],
        ).compute()
    assert str(exc_info.value) in [
        "Cannot store empty datasets",  # dask <= 2021.5.0
        "Cannot store empty datasets, partition_list must not be empty if in store mode.",  # dask > 2021.5.0 + shuffle == True
        "No data left to save outside partition columns",  # dask > 2021.5.0 + shuffle == False
    ]
