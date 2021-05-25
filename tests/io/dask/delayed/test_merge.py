import pickle

import dask
import pytest

from kartothek.io.dask.delayed import merge_datasets_as_delayed
from kartothek.io.testing.merge import *  # noqa


def _merge_datasets(*args, **kwargs):
    df_list = merge_datasets_as_delayed(*args, **kwargs)
    s = pickle.dumps(df_list, pickle.HIGHEST_PROTOCOL)
    df_list = pickle.loads(s)
    return dask.compute(df_list)[0]


@pytest.fixture
def bound_merge_datasets(request):
    return _merge_datasets
