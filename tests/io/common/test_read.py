import datetime
import pickle
from functools import partial
from itertools import permutations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from storefact import get_store_from_url

from kartothek.io.dask.bag import (
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
)
from kartothek.io.dask.dataframe import read_dataset_as_ddf
from kartothek.io.dask.delayed import read_dataset_as_delayed
from kartothek.io.eager import read_dataset_as_dataframes, read_table
from kartothek.io.iter import (
    read_dataset_as_dataframes__iterator,
    store_dataframes_as_dataset__iter,
)
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition


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


@pytest.fixture(params=[True, False], ids=["use_factory", "no_factory"])
def use_dataset_factory(request, dates_as_object):
    return request.param


@pytest.fixture(params=[True, False], ids=["dates_as_object", "datest_as_datetime"])
def dates_as_object(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["use_categoricals", "no_categoricals"])
def use_categoricals(request):
    return request.param


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
def test_read_dataset_as_dataframes_predicate(
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


@pytest.mark.parametrize(
    "predicates",
    [
        [[("P", "==", 2), ("TARGET", "==", 2)]],
        [[("P", "in", [2]), ("TARGET", "==", 2)]],
        [[("P", "!=", 1), ("L", "==", 2)]],
        [[("P", "!=", 1), ("L", "in", [2])]],
        [[("P", ">", 2)], [("TARGET", ">=", 2)]],
        [[("P", ">=", 2)], [("TARGET", ">=", 2)]],
    ],
)
def test_read_dataset_as_dataframes_predicate_with_partition_keys(
    dataset_partition_keys,
    store_session_factory,
    bound_load_dataframes,
    predicates,
    output_type,
):
    if output_type != "dataframe":
        pytest.skip()
    result = bound_load_dataframes(
        dataset_uuid=dataset_partition_keys.uuid,
        store=store_session_factory,
        predicates=predicates,
    )

    core_result = pd.concat(result)

    expected_core = pd.DataFrame(
        {"P": [2], "L": [2], "TARGET": [2], "DATE": [datetime.date(2009, 12, 31)]}
    )
    pdt.assert_frame_equal(
        core_result, expected_core, check_dtype=False, check_like=True
    )


def test_read_dataset_as_dataframes_predicate_empty(
    dataset_partition_keys, store_session_factory, output_type, bound_load_dataframes,
):
    if output_type != "dataframe":
        pytest.skip()
    result = bound_load_dataframes(
        dataset_uuid=dataset_partition_keys.uuid,
        store=store_session_factory,
        predicates=[[("P", "==", -42)]],
        columns={SINGLE_TABLE: ["P", "L", "TARGET"]},
    )
    assert len(result) == 0


def test_read_dataset_as_dataframes_dispatch_by_empty(
    store_session_factory,
    dataset_dispatch_by,
    bound_load_dataframes,
    backend_identifier,
    output_type,
    metadata_version,
    dataset_dispatch_by_uuid,
):
    if output_type == "table":
        pytest.skip()
    # Dispatch by primary index "A"
    dispatched = bound_load_dataframes(
        dataset_uuid=dataset_dispatch_by_uuid,
        store=store_session_factory,
        dispatch_by=[],
    )

    assert len(dispatched) == 1


@pytest.mark.parametrize("dispatch_by", ["A", "B", "C"])
def test_read_dataset_as_dataframes_dispatch_by_single_col(
    store_session_factory,
    dataset_dispatch_by,
    bound_load_dataframes,
    backend_identifier,
    dispatch_by,
    output_type,
    metadata_version,
    dataset_dispatch_by_uuid,
):
    if output_type == "table":
        pytest.skip()
    # Dispatch by primary index "A"
    dispatched_a = bound_load_dataframes(
        dataset_uuid=dataset_dispatch_by_uuid,
        store=store_session_factory,
        dispatch_by=[dispatch_by],
    )

    unique_a = set()
    for data in dispatched_a:
        unique_dispatch = data[dispatch_by].unique()
        assert len(unique_dispatch) == 1
        assert unique_dispatch[0] not in unique_a
        unique_a.add(unique_dispatch[0])


def test_read_dataset_as_dataframes_dispatch_by_multi_col(
    store_session_factory,
    bound_load_dataframes,
    output_type,
    dataset_dispatch_by,
    dataset_dispatch_by_uuid,
):
    if output_type == "table":
        pytest.skip()
    for dispatch_by in permutations(("A", "B", "C"), 2):
        dispatched = bound_load_dataframes(
            dataset_uuid=dataset_dispatch_by_uuid,
            store=store_session_factory,
            dispatch_by=dispatch_by,
        )
        uniques = pd.DataFrame(columns=dispatch_by)
        for part in dispatched:
            if isinstance(part, MetaPartition):
                data = part.data
            else:
                data = part
            unique_dispatch = data[list(dispatch_by)].drop_duplicates()
            assert len(unique_dispatch) == 1
            row = unique_dispatch
            uniques.append(row)
        assert not any(uniques.duplicated())


@pytest.mark.parametrize(
    "dispatch_by, predicates, expected_dispatches",
    [
        # This should only dispatch one partition since there is only
        # one file with valid data points
        (["A"], [[("C", ">", 2)]], 1),
        # We dispatch and restrict to one valie, i.e. one dispatch
        (["B"], [[("B", "==", 10)]], 1),
        # The same is true for a non-partition index col
        (["C"], [[("C", "==", 1)]], 1),
        # A condition where both primary and secondary indices need to work together
        (["A", "C"], [[("A", ">", 1), ("C", "<", 3)]], 2),
    ],
)
def test_read_dispatch_by_with_predicates(
    store_session_factory,
    dataset_dispatch_by_uuid,
    bound_load_dataframes,
    dataset_dispatch_by,
    dispatch_by,
    output_type,
    expected_dispatches,
    predicates,
):
    if output_type == "table":
        pytest.skip()

    dispatched = bound_load_dataframes(
        dataset_uuid=dataset_dispatch_by_uuid,
        store=store_session_factory,
        dispatch_by=dispatch_by,
        predicates=predicates,
    )

    assert len(dispatched) == expected_dispatches, dispatched


def _perform_read_test(
    dataset_uuid,
    store_factory,
    execute_read_callable,
    use_categoricals,
    output_type,
    dates_as_object,
    read_kwargs=None,
    ds_factory=None,
):
    if not read_kwargs:
        read_kwargs = {}
    if use_categoricals:
        # dataset_with_index has an index on L but not on P
        categoricals = ["P", "L"]
    else:
        categoricals = None

    result = execute_read_callable(
        dataset_uuid=dataset_uuid,
        store=store_factory,
        factory=ds_factory,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        **read_kwargs,
    )

    assert len(result) == 2

    if output_type == "metapartition":
        for res in result:
            assert isinstance(res, MetaPartition)
        result = [mp.data for mp in result]

        def sort_by(obj):
            return obj[SINGLE_TABLE].P.iloc[0]

    elif output_type == "table":
        assert isinstance(result[0], pd.DataFrame)
        assert "P" in result[0]

        def sort_by(obj):
            return obj.P.iloc[0]

    else:
        assert isinstance(result[0], pd.DataFrame)
        assert "P" in result[0]

        def sort_by(obj):
            return obj.P.iloc[0]

    result = sorted(result, key=sort_by)

    expected_df_core_1 = pd.DataFrame(
        {"P": [1], "L": [1], "TARGET": [1], "DATE": [datetime.date(2010, 1, 1)]}
    )
    expected_df_core_2 = pd.DataFrame(
        {"P": [2], "L": [2], "TARGET": [2], "DATE": [datetime.date(2009, 12, 31)]}
    )
    expected_dfs = [
        expected_df_core_1,
        expected_df_core_2,
    ]

    for res, expected_df_core in zip(result, expected_dfs):
        if not dates_as_object:
            expected_df_core["DATE"] = pd.to_datetime(expected_df_core["DATE"])
        if use_categoricals:
            expected_df_core = expected_df_core.astype(
                {"P": "category", "L": "category"}
            )

        pdt.assert_frame_equal(
            res.reset_index(drop=True),
            expected_df_core.reset_index(drop=True),
            check_dtype=False,
            check_like=True,
            check_categorical=False,
        )


def test_read_dataset_as_dataframes(
    dataset,
    store_session_factory,
    dataset_factory,
    use_dataset_factory,
    bound_load_dataframes,
    use_categoricals,
    output_type,
    dates_as_object,
):
    if use_dataset_factory:
        dataset_uuid = dataset.uuid
        store_factory = store_session_factory
        ds_factory = None
    else:
        dataset_uuid = None
        store_factory = None
        ds_factory = dataset_factory

    _perform_read_test(
        dataset_uuid=dataset_uuid,
        store_factory=store_factory,
        ds_factory=ds_factory,
        execute_read_callable=bound_load_dataframes,
        use_categoricals=use_categoricals,
        output_type=output_type,
        dates_as_object=dates_as_object,
    )


def test_store_input_types(store_input_types, bound_load_dataframes):
    from kartothek.io.eager import store_dataframes_as_dataset
    from kartothek.serialization.testing import get_dataframe_not_nested

    dataset_uuid = "dataset_uuid"
    df = get_dataframe_not_nested(10)

    store_dataframes_as_dataset(
        dfs=[df],
        dataset_uuid=dataset_uuid,
        store=store_input_types,
        partition_on=[df.columns[0]],
        secondary_indices=[df.columns[1]],
    )

    # Use predicates to trigger partition pruning with indices
    predicates = [
        [
            (df.columns[0], "==", df.loc[0, df.columns[0]]),
            (df.columns[1], "==", df.loc[0, df.columns[1]]),
        ]
    ]

    result = bound_load_dataframes(
        dataset_uuid=dataset_uuid,
        store=store_input_types,
        predicates=predicates,
        dates_as_object=True,
    )

    if isinstance(result, list):
        result = result[0]

    if isinstance(result, MetaPartition):
        result = result.data

    if isinstance(result, dict):
        result = result[SINGLE_TABLE]

    pdt.assert_frame_equal(result, df.head(1), check_dtype=False)
