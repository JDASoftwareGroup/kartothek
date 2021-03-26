import pickle

import dask
import dask.dataframe as dd
import pytest
from tests.io.common.conftest import update_dataset_dataframe

from kartothek.io.dask.dataframe import update_dataset_from_ddf


@pytest.fixture
def bound_update_dataset():
    return update_dataset_dataframe


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
    with pytest.raises(ValueError, match="Cannot store empty datasets"):
        update_dataset_from_ddf(
            dask.dataframe.from_delayed([], meta=(("a", int),)),
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=shuffle,
            partition_on=["a"],
        ).compute()
