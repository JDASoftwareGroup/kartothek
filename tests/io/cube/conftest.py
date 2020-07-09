import dask
import pytest


@pytest.fixture(params=["dask_bag_bs1", "dask_bag_bs3", "dask_dataframe", "eager"])
def driver_name(request):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def setup_dask():
    dask.config.set(scheduler="synchronous")


@pytest.fixture
def skip_eager(driver_name):
    if driver_name == "eager":
        pytest.skip("Skipped for eager backend.")
