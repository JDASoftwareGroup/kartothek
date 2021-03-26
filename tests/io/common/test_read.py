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
from kartothek.io.eager import (
    read_dataset_as_dataframes,
    read_table,
    store_dataframes_as_dataset,
)
from kartothek.io.iter import read_dataset_as_dataframes__iterator
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition


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


def test_read_dataset_as_dataframes_columns_projection(
    store_factory, bound_load_dataframes, metadata_version
):
    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return df

    in_partitions = [_f([1, 100])]
    dataset_uuid = "partitioned_uuid"
    store_dataframes_as_dataset(
        dfs=in_partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        partition_on=["a", "b"],
    )

    result = bound_load_dataframes(
        dataset_uuid=dataset_uuid, store=store_factory, columns=["a", "b", "c"],
    )
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)

    expected_df = pd.DataFrame({"a": [1, 1], "b": [1, 1], "c": [100, 100]})
    pdt.assert_frame_equal(expected_df, result_df, check_like=True)


def test_read_dataset_as_dataframes_columns_primary_index_only(
    store_factory, bound_load_dataframes, metadata_version
):
    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return df

    in_partitions = [_f([1, 100])]
    dataset_uuid = "partitioned_uuid"

    store_dataframes_as_dataset(
        dfs=in_partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        partition_on=["a", "b"],
    )
    result = bound_load_dataframes(
        dataset_uuid=dataset_uuid, store=store_factory, columns=["a", "b"]
    )
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)

    expected_df = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
    pdt.assert_frame_equal(expected_df, result_df, check_like=True)


def test_empty_predicate_pushdown_empty_col_projection(
    dataset, store_session_factory, bound_load_dataframes, backend_identifier
):
    result = bound_load_dataframes(
        dataset_uuid=dataset.uuid,
        store=store_session_factory,
        columns=[],
        predicates=[[("P", "==", 12345678)]],  # this product doesn't exist
    )

    if backend_identifier.startswith("dask"):
        pytest.xfail("Output of dask for empty results is currently inconsistent")
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    res = pd.concat(result_dfs).reset_index(drop=True)
    pdt.assert_frame_equal(res, pd.DataFrame(index=pd.RangeIndex(start=0, stop=0)))


@pytest.mark.parametrize("partition_on", [["a", "b"], ["c"], ["a", "b", "c"]])
@pytest.mark.parametrize("datetype", [datetime.datetime, datetime.date])
@pytest.mark.parametrize("comp", ["==", ">="])
def test_datetime_predicate_with_dates_as_object(
    dataset,
    store_factory,
    bound_load_dataframes,
    metadata_version,
    output_type,
    partition_on,
    datetype,
    comp,
):
    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return df

    in_partitions = [_f([1, datetype(2000, 1, 1)])]
    dataset_uuid = "partitioned_uuid"
    store_dataframes_as_dataset(
        dfs=in_partitions,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        partition_on=partition_on,
    )

    result = bound_load_dataframes(
        dataset_uuid="partitioned_uuid",
        store=store_factory,
        predicates=[[("c", comp, datetype(2000, 1, 1))]],
        dates_as_object=True,
    )
    if output_type != "dataframe":
        return

    assert len(result) == 1
    df_actual = result[0]

    df_expected = in_partitions[0]
    pdt.assert_frame_equal(df_actual, df_expected, check_like=True)


def test_binary_column_metadata(store_factory, bound_load_dataframes):
    df = pd.DataFrame({b"int_col": [1], "🙈".encode(): [2]})

    store_dataframes_as_dataset(
        dfs=[df], store=store_factory, dataset_uuid="dataset_uuid"
    )

    result = bound_load_dataframes(dataset_uuid="dataset_uuid", store=store_factory)

    probe = result[0]
    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    df = pd.concat(result_dfs).reset_index(drop=True)

    # Assert column names are of type `str`, instead of `bytes` objects
    assert set(df.columns.map(type)) == {str}


def test_extensiondtype_rountrip(store_factory, bound_load_dataframes):
    df = pd.DataFrame({"str": pd.Series(["a", "b"], dtype="string")})

    store_dataframes_as_dataset(
        dfs=[df], store=store_factory, dataset_uuid="dataset_uuid"
    )

    result = bound_load_dataframes(dataset_uuid="dataset_uuid", store=store_factory)

    probe = result[0]
    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)
    pdt.assert_frame_equal(df, result_df)


def test_non_default_table_name_roundtrip(store_factory, bound_load_dataframes):
    df = pd.DataFrame({"A": [1]})
    store_dataframes_as_dataset(
        dfs=[df], store=store_factory, dataset_uuid="dataset_uuid", table_name="foo"
    )
    result = bound_load_dataframes(dataset_uuid="dataset_uuid", store=store_factory)

    probe = result[0]
    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)
    pdt.assert_frame_equal(df, result_df)
