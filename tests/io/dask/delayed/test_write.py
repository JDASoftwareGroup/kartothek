from functools import partial

import dask.bag as db
import pytest

from kartothek.io.dask.bag import store_bag_as_dataset
from kartothek.io.dask.delayed import store_delayed_as_dataset
from kartothek.io.testing.write import *  # noqa


def _store_dataframes(execution_mode, df_list, *args, **kwargs):
    if execution_mode == "dask.bag":
        return store_bag_as_dataset(
            db.from_sequence(df_list), *args, **kwargs
        ).compute()
    elif execution_mode == "dask.delayed":
        return store_delayed_as_dataset(df_list, *args, **kwargs).compute()
    else:
        raise ValueError("Unknown execution mode: {}".format(execution_mode))


@pytest.fixture(params=["dask.delayed", "dask.bag"])
def bound_store_dataframes(request):
    return partial(_store_dataframes, request.param)
