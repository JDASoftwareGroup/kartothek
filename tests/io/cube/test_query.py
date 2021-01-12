import pytest
from tests.io.cube.utils import wrap_bag_read, wrap_ddf_read

from kartothek.io.dask.bag_cube import query_cube_bag
from kartothek.io.dask.dataframe_cube import query_cube_dataframe
from kartothek.io.eager_cube import query_cube
from kartothek.io.testing.query_cube import *  # noqa


@pytest.fixture(scope="session")
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_read(query_cube_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_read(query_cube_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        return wrap_ddf_read(query_cube_dataframe)
    elif driver_name == "eager":
        return query_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))
