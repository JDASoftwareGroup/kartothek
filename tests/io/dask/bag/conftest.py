import pytest


@pytest.fixture()
def backend_identifier():
    return "dask.bag"
