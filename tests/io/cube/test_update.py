import pytest
from tests.io.cube.utils import wrap_bag_write

from kartothek.io.dask.bag_cube import update_cube_from_bag
from kartothek.io.testing.update_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_write(update_cube_from_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_write(update_cube_from_bag, blocksize=3)
    else:
        pytest.skip()
