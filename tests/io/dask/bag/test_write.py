import pickle

import dask.bag as db
import pytest

from kartothek.io.dask.bag import store_bag_as_dataset
from kartothek.io.testing.write import *  # noqa


def _store_dataframes(df_list, *args, **kwargs):
    bag = store_bag_as_dataset(db.from_sequence(df_list), *args, **kwargs)
    s = pickle.dumps(bag, pickle.HIGHEST_PROTOCOL)
    bag = pickle.loads(s)
    return bag.compute()


@pytest.fixture()
def bound_store_dataframes(request):
    return _store_dataframes
