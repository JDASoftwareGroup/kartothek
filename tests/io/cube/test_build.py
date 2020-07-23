import uuid

import dask
import dask.bag as db
import dask.core
import pandas as pd
import pytest
from tests.io.cube.utils import wrap_bag_write, wrap_ddf_write

from kartothek.core.cube.cube import Cube
from kartothek.io.dask.bag_cube import build_cube_from_bag
from kartothek.io.dask.dataframe_cube import build_cube_from_dataframe
from kartothek.io.eager_cube import build_cube
from kartothek.io.testing.build_cube import *  # noqa


@pytest.fixture
def driver(driver_name):
    if driver_name == "dask_bag_bs1":
        return wrap_bag_write(build_cube_from_bag, blocksize=1)
    elif driver_name == "dask_bag_bs3":
        return wrap_bag_write(build_cube_from_bag, blocksize=3)
    elif driver_name == "dask_dataframe":
        return wrap_ddf_write(build_cube_from_dataframe)
    elif driver_name == "eager":
        return build_cube
    else:
        raise ValueError("Unknown driver: {}".format(driver_name))


def _count_execution_to_store(obj, store):
    store = store()
    key = "counter.{}".format(uuid.uuid4().hex)
    store.put(key, b"")
    return obj


def test_dask_bag_fusing(driver, function_store, driver_name, skip_eager):
    """
    There were two issues with the dask.bag write path.

    Ideal
    -----
    With 4 partitions and 2 datasets to write, it should look like this:

        o-+
          +
        o-+
          +-o
        o-+
          +
        o-+

    Missing linear fusing
    ---------------------
    The bags did not have linear fusing:

        o-o-o-o-o-+
                  +
        o-o-o-o-o-+
                  +-o
        o-o-o-o-o-+
                  +
        o-o-o-o-o-+

    Process-then-write instead of one-at-the-time
    ---------------------------------------------
    Due to the implementation of using 1 write bag per dataset and a pluck/split operation, the data for the whole bag
    partition was kept, then split, then written. Instead we aim for processing (including write) each DF in the
    partition and then move all metadata to the correct write path:

        o-s>-+
          v  |
          |  |
        o-------s>-+
          |  |  v  |
          |  |  |  |
        o-------------s>-+
          |  |  |  |  v  |
          |  |  |  |  |  |
        o-------------------s--+
          |  |  |  |  |  |  |  |
          +-----+-----+-----+-----o--+
             |     |     |     |     +-o
             +-----+-----+-----+--o--+
    """

    partition_size = 1 if driver_name == "dask_bag_bs1" else 3
    n_partitions = 4

    dfs = [
        {
            "source": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v1": 42}),
            "enrich": pd.DataFrame({"x": [2 * i, 2 * i + 1], "p": i, "v2": 1337}),
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
    bag = build_cube_from_bag(
        data=bag,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids=["source", "enrich"],
    )
    dct = dask.optimize(bag)[0].__dask_graph__()
    tasks = {k for k, v in dct.items() if dask.core.istask(v)}
    assert len(tasks) == (n_partitions + 1)


def test_function_executed_once(driver, function_store, driver_name, skip_eager):
    """
    Test that the payload function is only executed once per branch.

    This was a bug in the dask_bag backend.
    """
    if driver_name == "dask_dataframe":
        pytest.skip("not relevant for dask.dataframe")

    df_source1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v1": [10, 11]})
    df_source2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v1": [12, 13]})
    df_enrich1 = pd.DataFrame({"x": [0, 1], "p": [0, 0], "v2": [20, 21]})
    df_enrich2 = pd.DataFrame({"x": [2, 3], "p": [1, 1], "v2": [22, 23]})

    dfs = [
        {"source": df_source1, "enrich": df_enrich1},
        {"source": df_source2, "enrich": df_enrich2},
    ]

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
        bag = build_cube_from_bag(
            data=bag,
            cube=cube,
            store=function_store,
            ktk_cube_dataset_ids=["source", "enrich"],
        )
        bag.compute()
    else:
        raise ValueError("Missing implementation for driver: {}".format(driver_name))

    assert len(function_store().keys(prefix="counter.")) == 2
