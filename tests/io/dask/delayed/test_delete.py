import pickle

import dask
import pytest

from kartothek.io.dask.delayed import (
    delete_dataset__delayed,
    garbage_collect_dataset__delayed,
)
from kartothek.io.testing.delete import *  # noqa


def _delete(*args, **kwargs):
    tasks = delete_dataset__delayed(*args, **kwargs)
    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)
    dask.compute(tasks)


@pytest.fixture
def bound_delete_dataset():
    return _delete


@pytest.fixture()
def garbage_collect_callable():
    return garbage_collect_dataset__delayed
