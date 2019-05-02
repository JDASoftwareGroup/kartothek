import dask
import pytest

from kartothek.io.dask.delayed import garbage_collect_dataset__delayed
from kartothek.io.testing.gc import *  # noqa


def _run_garbage_collect(*args, **kwargs):
    tasks = garbage_collect_dataset__delayed(*args, **kwargs)
    dask.compute(tasks)


@pytest.fixture()
def garbage_collect_callable():
    return _run_garbage_collect
