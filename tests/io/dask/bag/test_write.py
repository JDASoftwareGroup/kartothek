import dask.bag as db
import pytest

from kartothek.io.dask.bag import store_bag_as_dataset
from kartothek.io.testing.write import *  # noqa


def _store_dataframes(df_list, *args, **kwargs):
    return store_bag_as_dataset(db.from_sequence(df_list), *args, **kwargs).compute()


@pytest.fixture()
def bound_store_dataframes(request):
    return _store_dataframes
