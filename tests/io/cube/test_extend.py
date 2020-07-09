import uuid

import dask
import dask.bag as db
import dask.core
import pandas as pd
import pytest
from tests.io.cube.utils import wrap_bag_write, wrap_ddf_write

from kartothek.core.cube.cube import Cube
from kartothek.io.dask.bag_cube import extend_cube_from_bag
from kartothek.io.dask.dataframe_cube import extend_cube_from_dataframe
from kartothek.io.eager_cube import extend_cube
from kartothek.io.testing.extend_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_write(extend_cube_from_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_write(extend_cube_from_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        return wrap_ddf_write(extend_cube_from_dataframe)
    elif driver_name == "eager":
        return extend_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))


def _count_execution_to_store(obj, store):
    store = store()
    key = "counter.{}".format(uuid.uuid4().hex)
    store.put(key, b"")
    return obj


def test_dask_bag_fusing(
    driver, function_store, driver_name, skip_eager, existing_cube
):
    """
    See kartothek/tests/io/cube/test_build.py::test_dask_bag_fusing
    """
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    partition_size = 1 if driver_name == "dask_bag_bs1" else 3
    n_partitions = 4

    dfs = [
        {
            "a": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v3": 42}),
            "b": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v4": 1337}),
        }
        for i in range(partition_size * n_partitions)
    ]

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    bag = db.from_sequence(dfs, partition_size=partition_size).map(
        _count_execution_to_store, store=function_store
    )
    bag = extend_cube_from_bag(
        data=bag, cube=cube, store=function_store, ktk_cube_dataset_ids=["a", "b"]
    )
    dct = dask.optimize(bag)[0].__dask_graph__()
    tasks = {k for k, v in dct.items() if dask.core.istask(v)}
    assert len(tasks) == (n_partitions + 1)


def test_function_executed_once(driver, function_store, driver_name, existing_cube):
    """
    Test that the payload function is only executed once per branch.

    This was a bug in the dask_bag backend.
    """
    if driver_name == "eager":
        pytest.skip("not relevant for eager")
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    df_a1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v3": [10, 11]})
    df_a2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v3": [12, 13]})
    df_b1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v4": [20, 21]})
    df_b2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v4": [22, 23]})

    dfs = [{"a": df_a1, "b": df_b1}, {"a": df_a2, "b": df_b2}]

    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
    )

    if driver_name in ("dask_bag_bs1", "dask_bag_bs3"):
        bag = db.from_sequence(
            dfs, partition_size=1 if driver_name == "dask_bag_bs1" else 3
        ).map(_count_execution_to_store, store=function_store)
        bag = extend_cube_from_bag(
            data=bag, cube=cube, store=function_store, ktk_cube_dataset_ids=["a", "b"]
        )
        bag.compute()
    else:
        raise ValueError("Missing implementation for driver: {}".format(driver_name))

    assert len(function_store().keys(prefix="counter.")) == 2
