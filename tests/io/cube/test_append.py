import pytest
from tests.io.cube.utils import wrap_bag_write, wrap_ddf_write

from kartothek.io.dask.bag_cube import append_to_cube_from_bag
from kartothek.io.dask.dataframe_cube import append_to_cube_from_dataframe
from kartothek.io.eager_cube import append_to_cube
from kartothek.io.testing.append_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_write(append_to_cube_from_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_write(append_to_cube_from_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        return wrap_ddf_write(append_to_cube_from_dataframe)
    elif driver_name == "eager":
        return append_to_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))
