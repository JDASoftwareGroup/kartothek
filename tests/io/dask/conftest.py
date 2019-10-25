import contextlib

import distributed
import distributed.utils_test
import pytest
from distributed import Client

import kartothek.core._time
from kartothek.core.testing import cm_frozen_time

_client = None


@pytest.fixture(autouse=True, scope="session")
def setup_dask_distributed():
    """
    This fixture makes all dask tests effectively use distributed under the hood.
    """
    global _client
    with distributed.utils_test.cluster() as (scheduler, workers):
        _client = Client(scheduler["address"])
        yield


@contextlib.contextmanager
def cm_distributed_frozen_time():
    global _client
    assert _client is not None
    _client.run(_freeze_time_on_worker, kartothek.core._time.datetime_now())
    try:
        yield
    finally:
        _client.run(_unfreeze_time_on_worker)


@pytest.fixture
def distributed_frozen_time(frozen_time):
    with cm_distributed_frozen_time():
        yield


@pytest.fixture
def frozen_time_em(frozen_time):
    with cm_distributed_frozen_time():
        yield


def _freeze_time_on_worker(freeze_time):
    # this runs on the distributed worker, initiated by cm_distributed_frozen_time
    # it runs the "enter" part of the cm_frozen_time context manager.
    # It saves the context manager to be able to call __exit__ on it later
    # in _unfreeze_time_on_worker.
    kartothek.core._time._datetime_utcnow_orig = kartothek.core._time.datetime_utcnow
    cm = cm_frozen_time(freeze_time)
    kartothek.core._time._time_patcher = cm
    cm.__enter__()


def _unfreeze_time_on_worker():
    kartothek.core._time._time_patcher.__exit__(None, None, None)
