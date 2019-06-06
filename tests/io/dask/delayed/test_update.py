import pickle

import dask
import pytest

from kartothek.io.dask.delayed import update_dataset_from_delayed
from kartothek.io.testing.update import *  # noqa


@pytest.fixture
def bound_update_dataset():
    return _update_dataset


@dask.delayed
def _unwrap_partition(part):
    return next(iter(dict(part["data"]).values()))


def _update_dataset(partitions, *args, **kwargs):
    tasks = update_dataset_from_delayed(partitions, *args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    return tasks.compute()
