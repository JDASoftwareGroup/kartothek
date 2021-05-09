import dask
import pytest
from tests.io.common.conftest import update_dataset_dataframe

from kartothek.io.dask.dataframe import update_dataset_from_ddf


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
