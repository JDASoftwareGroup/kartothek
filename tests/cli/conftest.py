import gc
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import storefact
from click.testing import CliRunner
from freezegun import freeze_time

from kartothek.cli import cli as f_cli
from kartothek.core.cube.cube import Cube
from kartothek.io.eager_cube import build_cube, extend_cube


@pytest.fixture
def runner():
    return CliRunner(echo_stdin=True)


@pytest.fixture(autouse=True)
def isolated(runner):
    with runner.isolated_filesystem():
        yield


@pytest.fixture
def cli(runner):
    def _cli(*args, **kwargs):
        input = kwargs.pop("input", None)
        return runner.invoke(f_cli, args, catch_exceptions=False, input=input)

    return _cli


@pytest.fixture
def cli_keep_stdio():
    """
    Like `cli` but do not redirect stdio. Useful mainly for interactive testing /
    embedding in tests while developing tests.
    """

    def _cli(*args, **kwargs):
        f_cli.main(args, **kwargs)

    return _cli


@pytest.fixture
def storecfg():
    path = "data"
    os.mkdir(path)

    return {"type": "filesystem", "path": path}


@pytest.fixture
def storecfg2():
    path = "data2"
    os.mkdir(path)

    return {"type": "filesystem", "path": path}


@pytest.fixture
def azurestorecfg(azure_store_cfg_factory):
    cfg = azure_store_cfg_factory("cli")
    store = storefact.get_store(**cfg)
    for k in list(store.keys()):
        store.delete(k)
    return cfg


@pytest.fixture
def azurestorecfg2(azure_store_cfg_factory):
    cfg = azure_store_cfg_factory("cli2")
    store = storefact.get_store(**cfg)
    for k in list(store.keys()):
        store.delete(k)
    return cfg


@pytest.fixture
def store(storecfg):
    return storefact.get_store(**storecfg)


@pytest.fixture
def store2(storecfg2):
    return storefact.get_store(**storecfg2)


@pytest.fixture
def azurestore(azurestorecfg):
    store = storefact.get_store(**azurestorecfg)
    yield store
    # prevent ResourceWarning
    gc.collect()
    store.block_blob_service.request_session.close()


@pytest.fixture
def azurestore2(azurestorecfg2):
    store = storefact.get_store(**azurestorecfg2)
    yield store
    # prevent ResourceWarning
    gc.collect()
    store.block_blob_service.request_session.close()


@pytest.fixture
def skv(storecfg, storecfg2):
    with open("skv.yml", "w") as fp:
        json.dump({"cubes": storecfg, "cubes2": storecfg2}, fp)


@pytest.fixture
def skv_azure(azurestorecfg, azurestorecfg2):
    with open("skv.yml", "w") as fp:
        json.dump({"cubes": azurestorecfg, "cubes2": azurestorecfg2}, fp)


@pytest.fixture
def df_source():
    return pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "p": 0,
            "q": ["a", "a", "b", "b"],
            "i1": [True, False, False, False],
            "v1": [13.0, 42.0, 13.37, 100.0],
        }
    )


@pytest.fixture
def df_enrich():
    return pd.DataFrame(
        {
            "y": [0, 1],
            "part": 0,
            "q": ["a", "b"],
            "i2": [pd.Timestamp("2018"), pd.Timestamp("2019")],
            "v2": [np.arange(10), np.arange(20)],
        }
    )


@pytest.fixture
def df_complete(df_source, df_enrich):
    df = df_source.merge(df_enrich).sort_values(["x", "y"]).reset_index(drop=True)
    return df.reindex(columns=sorted(df.columns))


@pytest.fixture
def cube(store, df_source, df_enrich):
    return Cube(
        uuid_prefix="my_cube",
        index_columns=["i1", "i2"],
        dimension_columns=["x", "y"],
        partition_columns=["p", "q"],
        seed_dataset="source",
    )


@pytest.fixture
def built_cube(store, cube, df_source, df_enrich):
    with freeze_time(datetime(2018, 1, 31, 14, 3, 22)):
        build_cube(data={"source": df_source}, cube=cube, store=store)

    with freeze_time(datetime(2019, 2, 28, 13, 1, 17)):
        extend_cube(
            data={"enrich": df_enrich},
            cube=cube,
            store=store,
            partition_on={"enrich": ["part", "q"]},
        )
    return cube


@pytest.fixture
def built_cube_azure(azurestore, cube, df_source):
    build_cube(data={"source": df_source}, cube=cube, store=azurestore)
    return cube


@pytest.fixture
@freeze_time(datetime(2019, 1, 31, 14, 3, 22))
def built_cube2(store2, cube, df_source):
    build_cube(data={"source": df_source}, cube=cube, store=store2)
