# -*- coding: utf-8 -*-
import pytest
from tests.io.cube.utils import wrap_bag_copy

from kartothek.io.dask.bag_cube import copy_cube_bag
from kartothek.io.eager_cube import copy_cube
from kartothek.io.testing.copy_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_copy(copy_cube_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_copy(copy_cube_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        pytest.skip("not supported for dask.dataframe")
    elif driver_name == "eager":
        return copy_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))
