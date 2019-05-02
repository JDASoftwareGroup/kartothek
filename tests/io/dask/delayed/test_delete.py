import dask
import pytest

from kartothek.io.dask.delayed import delete_dataset__delayed
from kartothek.io.testing.delete import *  # noqa


def _delete(*args, **kwargs):
    tasks = delete_dataset__delayed(*args, **kwargs)
    dask.compute(tasks)


@pytest.fixture
def bound_delete_dataset():
    return _delete
