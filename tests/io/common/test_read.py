import datetime
import pickle
from functools import partial

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.io.dask.bag import (
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
)
from kartothek.io.dask.dataframe import read_dataset_as_ddf
from kartothek.io.dask.delayed import read_dataset_as_delayed
from kartothek.io.eager import read_dataset_as_dataframes, read_table
from kartothek.io.iter import read_dataset_as_dataframes__iterator
from kartothek.io_components.metapartition import SINGLE_TABLE


@pytest.fixture(params=["eager", "iter", "bag", "delayed", "dataframe"])
def implementation_type(request):
    return request.param


@pytest.fixture(params=["dataframe", "table"])
def output_type(request, implementation_type):
    if (implementation_type in ["iter", "bag", "delayed"]) and (
        request.param == "table"
    ):
        pytest.skip()
    if (implementation_type == "dataframe") and (request.param == "dataframe"):
        pytest.skip()
    return request.param


def _read_table(*args, **kwargs):
    kwargs.pop("dispatch_by", None)
    res = read_table(*args, **kwargs)

    if len(res):
        # Array split conserves dtypes
        return np.array_split(res, len(res))
    else:
        return [res]


def _load_dataframes_iter(output_type, *args, **kwargs):
    if output_type == "dataframe":
        func = read_dataset_as_dataframes__iterator
    else:
        raise ValueError("Unknown output type {}".format(output_type))
    return list(func(*args, **kwargs))


# FIXME: handle removal of metparittion function properly.
# FIXME: consolidate read_Dataset_as_dataframes (replaced by iter)
def _read_dataset_eager(output_type, *args, **kwargs):
    if output_type == "table":
        return _read_table
    elif output_type == "dataframe":
        return read_dataset_as_dataframes
    else:
        raise NotImplementedError()


def _load_dataframes_bag(output_type, *args, **kwargs):
    if output_type == "dataframe":
        func = read_dataset_as_dataframe_bag
    elif output_type == "metapartition":
        func = read_dataset_as_metapartitions_bag
    tasks = func(*args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    result = tasks.compute()
    return result


def _load_dataframes_delayed(output_type, *args, **kwargs):
    if "tables" in kwargs:
        param_tables = kwargs.pop("tables")
        kwargs["table"] = param_tables
    func = partial(read_dataset_as_delayed)
    tasks = func(*args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    result = [task.compute() for task in tasks]
    return result


def _read_as_ddf(
    dataset_uuid,
    store,
    factory=None,
    categoricals=None,
    tables=None,
    dataset_has_index=False,
    **kwargs,
):
    table = tables or SINGLE_TABLE

    ddf = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        categoricals=categoricals,
        table=table,
        **kwargs,
    )
    if categoricals:
        assert ddf._meta.dtypes["P"] == pd.api.types.CategoricalDtype(
            categories=["__UNKNOWN_CATEGORIES__"], ordered=False
        )
        if dataset_has_index:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=[1, 2], ordered=False
            )
        else:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=["__UNKNOWN_CATEGORIES__"], ordered=False
            )

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    ddf = ddf.compute().reset_index(drop=True)

    def extract_dataframe(ix):
        df = ddf.iloc[[ix]].copy()
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.remove_unused_categories()
        return df.reset_index(drop=True)

    return [extract_dataframe(ix) for ix in ddf.index]


@pytest.fixture()
def bound_load_dataframes(output_type, implementation_type):
    if implementation_type == "eager":
        return _read_dataset_eager(output_type)
    elif implementation_type == "iter":
        return partial(_load_dataframes_iter, output_type)
    elif implementation_type == "bag":
        return partial(_load_dataframes_bag, output_type)
    elif implementation_type == "delayed":
        return partial(_load_dataframes_delayed, output_type)
    elif implementation_type == "dataframe":
        return _read_as_ddf
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "predicates",
    [
        [[("P", "==", 2)]],
        [[("P", "in", [2])]],
        [[("P", "!=", 1)]],
        [[("P", ">", 1)]],
        [[("P", ">=", 2)]],
    ],
)
def test_read_dataset_as_dataframes_predicate_base(
    dataset, store_session_factory, bound_load_dataframes, predicates, output_type
):
    if output_type != "dataframe":
        pytest.skip()
    result = bound_load_dataframes(
        dataset_uuid=dataset.uuid, store=store_session_factory, predicates=predicates,
    )
    core_result = pd.concat(result)

    expected_core = pd.DataFrame(
        {"P": [2], "L": [2], "TARGET": [2], "DATE": [datetime.date(2009, 12, 31)]}
    )
    pdt.assert_frame_equal(
        core_result, expected_core, check_dtype=False, check_like=True
    )
