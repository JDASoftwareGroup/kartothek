import pickle
from functools import partial

import dask
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from storefact import get_store_from_url

from kartothek.io.dask.bag import (
    build_dataset_indices__bag,
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
    store_bag_as_dataset,
)
from kartothek.io.dask.dataframe import read_dataset_as_ddf, update_dataset_from_ddf
from kartothek.io.dask.delayed import (
    delete_dataset__delayed,
    read_dataset_as_delayed,
    store_delayed_as_dataset,
    update_dataset_from_delayed,
)
from kartothek.io.eager import (
    build_dataset_indices,
    delete_dataset,
    read_dataset_as_dataframes,
    read_table,
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
)
from kartothek.io.iter import (
    read_dataset_as_dataframes__iterator,
    store_dataframes_as_dataset__iter,
    update_dataset_from_dataframes__iter,
)
from kartothek.io_components.metapartition import SINGLE_TABLE


class NoPickle:
    def __getstate__(self):
        raise RuntimeError("do NOT pickle this object!")


def mark_nopickle(obj):
    setattr(obj, "_nopickle", NoPickle())


def no_pickle_factory(url):
    return partial(no_pickle_store, url)


def no_pickle_store(url):
    store = get_store_from_url(url)
    mark_nopickle(store)
    return store


@pytest.fixture(params=["URL", "KeyValue", "Callable"])
def store_input_types(request, tmpdir):
    url = f"hfs://{tmpdir}"

    if request.param == "URL":
        return url
    elif request.param == "KeyValue":
        return get_store_from_url(url)
    elif request.param == "Callable":
        return no_pickle_factory(url)
    else:
        raise RuntimeError(f"Encountered unknown store type {type(request.param)}")


@pytest.fixture(params=["eager", "iter", "dask.bag", "dask.delayed", "dask.dataframe"])
def backend_identifier(request):
    return request.param


@pytest.fixture(params=["dataframe", "table"])
def output_type(request, backend_identifier):
    if (backend_identifier in ["iter", "dask.bag", "dask.delayed"]) and (
        request.param == "table"
    ):
        pytest.skip()
    if (backend_identifier == "dask.dataframe") and (request.param == "dataframe"):
        pytest.skip()
    return request.param


@pytest.fixture(scope="session")
def dataset_dispatch_by_uuid():
    import uuid

    return uuid.uuid1().hex


@pytest.fixture(scope="session")
def dataset_dispatch_by(
    metadata_version, store_session_factory, dataset_dispatch_by_uuid
):
    cluster1 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [1, 2], "Content": ["cluster1", "cluster1"]}
    )
    cluster2 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [2, 3], "Content": ["cluster2", "cluster2"]}
    )
    cluster3 = pd.DataFrame({"A": [1], "B": [20], "C": [1], "Content": ["cluster3"]})
    cluster4 = pd.DataFrame(
        {"A": [2, 2], "B": [10, 10], "C": [1, 2], "Content": ["cluster4", "cluster4"]}
    )
    clusters = [cluster1, cluster2, cluster3, cluster4]

    store_dataframes_as_dataset__iter(
        df_generator=clusters,
        store=store_session_factory,
        dataset_uuid=dataset_dispatch_by_uuid,
        metadata_version=metadata_version,
        partition_on=["A", "B"],
        secondary_indices=["C"],
    )
    return pd.concat(clusters).sort_values(["A", "B", "C"]).reset_index(drop=True)


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
def bound_load_dataframes(output_type, backend_identifier):
    if backend_identifier == "eager":
        return _read_dataset_eager(output_type)
    elif backend_identifier == "iter":
        return partial(_load_dataframes_iter, output_type)
    elif backend_identifier == "dask.bag":
        return partial(_load_dataframes_bag, output_type)
    elif backend_identifier == "dask.delayed":
        return partial(_load_dataframes_delayed, output_type)
    elif backend_identifier == "dask.dataframe":
        return _read_as_ddf
    else:
        raise NotImplementedError


def store_dataframes_eager(dfs, **kwargs):
    # Positional arguments in function but `None` is acceptable input
    for kw in ("dataset_uuid", "store"):
        if kw not in kwargs:
            kwargs[kw] = None

    return store_dataframes_as_dataset(dfs=dfs, **kwargs)


def _store_dataframes_iter(df_list, *args, **kwargs):
    df_generator = (x for x in df_list)
    return store_dataframes_as_dataset__iter(df_generator, *args, **kwargs)


def _store_dataframes_dask_bag(df_list, *args, **kwargs):
    bag = store_bag_as_dataset(db.from_sequence(df_list), *args, **kwargs)
    s = pickle.dumps(bag, pickle.HIGHEST_PROTOCOL)
    bag = pickle.loads(s)
    return bag.compute()


def _store_dataframes_dask_delayed(df_list, *args, **kwargs):
    tasks = store_delayed_as_dataset(df_list, *args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    return tasks.compute()


@pytest.fixture()
def bound_store_dataframes(backend_identifier):
    if backend_identifier == "eager":
        return store_dataframes_eager
    elif backend_identifier == "iter":
        return _store_dataframes_iter
    elif backend_identifier == "dask.bag":
        return _store_dataframes_dask_bag
    elif backend_identifier == "dask.delayed":
        return _store_dataframes_dask_delayed
    elif backend_identifier == "dask.dataframe":
        # not implemented for dask.dataframe
        pytest.skip()
    else:
        raise NotImplementedError


def _update_dataset_iter(df_list, *args, **kwargs):
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    df_generator = (x for x in df_list)
    return update_dataset_from_dataframes__iter(df_generator, *args, **kwargs)


def _update_dataset_delayed(partitions, *args, **kwargs):
    if not isinstance(partitions, list):
        partitions = [partitions]
    tasks = update_dataset_from_delayed(partitions, *args, **kwargs)

    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)

    return tasks.compute()


def _id(part):
    if isinstance(part, pd.DataFrame):
        return part
    else:
        return part[0]


def update_dataset_dataframe(partitions, *args, **kwargs):
    # TODO: Simplify once parse_input_to_metapartition is removed / obsolete

    if isinstance(partitions, pd.DataFrame):
        partitions = dd.from_pandas(partitions, npartitions=1)
    elif partitions is not None:
        delayed_partitions = [dask.delayed(_id)(part) for part in partitions]
        partitions = dd.from_delayed(delayed_partitions)
    else:
        partitions = None

    ddf = update_dataset_from_ddf(partitions, *args, **kwargs)

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    return ddf.compute()


def _return_none():
    return None


@pytest.fixture()
def bound_update_dataset(backend_identifier):
    if backend_identifier == "eager":
        return update_dataset_from_dataframes
    elif backend_identifier == "iter":
        return _update_dataset_iter
    elif backend_identifier == "dask.bag":
        # no tests impleme ted for update and dask.bag
        pytest.skip()
    elif backend_identifier == "dask.delayed":
        return _update_dataset_delayed
    elif backend_identifier == "dask.dataframe":
        return update_dataset_dataframe
    else:
        raise NotImplementedError


def _delete_delayed(*args, **kwargs):
    tasks = delete_dataset__delayed(*args, **kwargs)
    s = pickle.dumps(tasks, pickle.HIGHEST_PROTOCOL)
    tasks = pickle.loads(s)
    dask.compute(tasks)


@pytest.fixture()
def bound_delete_dataset(backend_identifier):
    if backend_identifier == "eager":
        return delete_dataset
    elif backend_identifier == "iter":
        # no tests implemented for delete and iter
        pytest.skip()
    elif backend_identifier == "dask.bag":
        # no tests implemented for update and dask.bag
        pytest.skip()
    elif backend_identifier == "dask.delayed":
        return _delete_delayed
    elif backend_identifier == "dask.dataframe":
        # no tests implemented for update and dask.dataframe
        pytest.skip()
    else:
        raise NotImplementedError


def _build_indices_bag(*args, **kwargs):
    bag = build_dataset_indices__bag(*args, **kwargs)

    # pickle roundtrip to ensure we don't need the inefficient cloudpickle fallback
    s = pickle.dumps(bag, pickle.HIGHEST_PROTOCOL)
    bag = pickle.loads(s)

    bag.compute()


@pytest.fixture()
def bound_build_dataset_indices(backend_identifier):
    if backend_identifier == "eager":
        return build_dataset_indices
    elif backend_identifier == "iter":
        # no tests implemented for index and iter
        pytest.skip()
    elif backend_identifier == "dask.bag":
        return _build_indices_bag
    elif backend_identifier == "dask.delayed":
        # no tests implemented for index and dask.delayed
        pytest.skip()
    elif backend_identifier == "dask.dataframe":
        # no tests implemented for index and dask.dataframe
        pytest.skip()
    else:
        raise NotImplementedError
