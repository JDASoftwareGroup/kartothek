import pytest
from tests.io.cube.utils import wrap_bag_delete

from kartothek.io.dask.bag_cube import delete_cube_bag
from kartothek.io.eager_cube import delete_cube
from kartothek.io.testing.delete_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_delete(delete_cube_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_delete(delete_cube_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        pytest.skip("not supported for dask.dataframe")
    elif driver_name == "eager":
        return delete_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))
