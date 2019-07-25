"""
This module is a collection of tests which should be implemented by all kartothek
**read** backends. The tests are not subject to the semantic versioning scheme and may change with minor or even patch releases.

To use the tests of this module, add the following import statement to your test module and ensure that the following fixtures are available in your test environment.

```
from kartothek.io.testing.read import *  # noqa
```

Fixtures required to be implemented:

* ``output_type`` - One of {`dataframe`, `metpartition`, `table`} to define the outptu type of the returned result.
* ``bound_load_dataframes`` - A callable which will retrieve the partitions in the format specified by ``output_type``. The callable should accept all keyword arguments expected for a kartothek reader.

Source test data

* ``dataset`` - A fixture generating test data (TODO: Expose this as a testing function)
* ``store_factory`` - A function scoped store factory
* ``store_session_factory`` - A session scoped store factory

Feature toggles (optional):

* ``custom_read_parameters`` - Pass additional backend specific kwargs to the read function. The fixture should return a dict which can be passed using the double asterisks syntax to the callable.

The following fixtures should be present (see tests.read.conftest)
* ``use_categoricals`` - Whether or not the call retrievs categorical data.
* ``dates_as_object`` - Whether or not the call retrievs date columns as objects.
* ``label_filter`` - a callable to filter partitions by label.

"""


import datetime
from functools import wraps
from itertools import permutations

import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io.iter import store_dataframes_as_dataset__iter
from kartothek.io_components.metapartition import MetaPartition


@pytest.fixture(
    params=["dataframe", "metapartition", "table"],
    ids=["dataframe", "metapartition", "table"],
)
def output_type(request):
    # TODO: get rid of this parametrization and split properly into two functions
    return request.param


@pytest.fixture(params=[True, False], ids=["use_categoricals", "no_categoricals"])
def use_categoricals(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["dates_as_object", "datest_as_datetime"])
def dates_as_object(request):
    return request.param


@pytest.fixture(
    params=[True, False],
    ids=["load_dataset_metadata_TRUE", "load_dataset_metadata_FALSE"],
)
def load_dataset_metadata(request):
    return request.param


@pytest.fixture(params=[None, lambda part_label: "cluster_1" in part_label])
def label_filter(request):
    return request.param


@pytest.fixture
def custom_read_parameters():
    return {}


@pytest.fixture(params=[True, False], ids=["use_factory", "no_factory"])
def use_dataset_factory(request, dates_as_object):
    return request.param


@pytest.fixture()
def bound_load_metapartitions():
    raise pytest.skip("Needs to implemented by submodule")


@pytest.fixture()
def bound_load_dataframes(bound_load_metapartitions):
    @wraps(bound_load_metapartitions)
    def _inner(*args, **kwargs):
        mps = bound_load_metapartitions(*args, **kwargs)
        return [mp.data for mp in mps]

    return _inner


def _strip_unused_categoricals(df):
    for col in df.columns:
        if pd.api.types.is_categorical(df[col]):
            df[col] = df[col].cat.remove_unused_categories()
    return df


def _perform_read_test(
    dataset_uuid,
    store_factory,
    execute_read_callable,
    use_categoricals,
    output_type,
    label_filter,
    dates_as_object,
    read_kwargs=None,
    ds_factory=None,
):
    if not read_kwargs:
        read_kwargs = {}
    if use_categoricals:
        # dataset_with_index has an index on L but not on P
        categoricals = {"core": ["P", "L"]}
    else:
        categoricals = None

    result = execute_read_callable(
        dataset_uuid=dataset_uuid,
        store=store_factory,
        factory=ds_factory,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        **read_kwargs,
    )

    # The filter should allow only a single partition
    if not label_filter:
        assert len(result) == 2
    else:
        # The filter should allow only a single partition
        assert len(result) == 1

    if output_type == "metapartition":
        for res in result:
            assert isinstance(res, MetaPartition)
        result = [mp.data for mp in result]

        def sort_by(obj):
            return obj["core"].P.iloc[0]

    elif output_type == "table":

        assert isinstance(result[0], pd.DataFrame)
        assert "P" in result[0]

        def sort_by(obj):
            return obj.P.iloc[0]

    else:
        assert isinstance(result[0], dict)
        assert "core" in result[0]
        assert "P" in result[0]["core"]

        def sort_by(obj):
            return obj["core"].P.iloc[0]

    result = sorted(result, key=sort_by)

    expected_df_core_1 = pd.DataFrame(
        {"P": [1], "L": [1], "TARGET": [1], "DATE": [datetime.date(2010, 1, 1)]}
    )
    expected_df_helper_1 = pd.DataFrame({"P": [1], "info": "a"})
    expected_df_core_2 = pd.DataFrame(
        {"P": [2], "L": [2], "TARGET": [2], "DATE": [datetime.date(2009, 12, 31)]}
    )
    expected_df_helper_2 = pd.DataFrame({"P": [2], "info": "b"})

    expected_dfs = [
        (expected_df_core_1, expected_df_helper_1),
        (expected_df_core_2, expected_df_helper_2),
    ]

    for res, (expected_df_core, expected_df_helper) in zip(result, expected_dfs):
        if not dates_as_object:
            expected_df_core["DATE"] = pd.to_datetime(expected_df_core["DATE"])
        if use_categoricals:
            expected_df_core = expected_df_core.astype(
                {"P": "category", "L": "category"}
            )

        if output_type == "table":

            pdt.assert_frame_equal(
                _strip_unused_categoricals(res).reset_index(drop=True),
                expected_df_core.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
        else:
            actual_core = _strip_unused_categoricals(res["core"])
            actual_helper = _strip_unused_categoricals(res["helper"])
            assert len(res) == 2
            pdt.assert_frame_equal(
                actual_core, expected_df_core, check_dtype=False, check_like=True
            )
            pdt.assert_frame_equal(
                actual_helper, expected_df_helper, check_dtype=False, check_like=True
            )


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
    dataset,
    store_session_factory,
    custom_read_parameters,
    bound_load_dataframes,
    predicates,
    output_type,
    backend_identifier,
):
    if output_type != "dataframe":
        pytest.skip()
    result = bound_load_dataframes(
        dataset_uuid=dataset.uuid,
        store=store_session_factory,
        predicates=predicates,
        **custom_read_parameters,
    )
    core_result = pd.concat([data["core"] for data in result])

    expected_core = pd.DataFrame(
        {
            "P": [2],
            "L": [2],
            "TARGET": [2],
            "DATE": pd.to_datetime([datetime.date(2009, 12, 31)]),
        }
    )
    pdt.assert_frame_equal(
        core_result, expected_core, check_dtype=False, check_like=True
    )
    helper_result = pd.concat([data["helper"] for data in result])
    expected_helper = pd.DataFrame({"P": [2], "info": "b"})
    pdt.assert_frame_equal(
        helper_result, expected_helper, check_dtype=False, check_like=True
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
    custom_read_parameters,
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
        tables=["core"],
        **custom_read_parameters,
    )

    core_result = pd.concat([data["core"] for data in result])

    expected_core = pd.DataFrame(
        {
            "P": [2],
            "L": [2],
            "TARGET": [2],
            "DATE": pd.to_datetime([datetime.date(2009, 12, 31)]),
        }
    )
    pdt.assert_frame_equal(
        core_result, expected_core, check_dtype=False, check_like=True
    )


def test_read_dataset_as_dataframes_predicate_empty(
    dataset_partition_keys,
    store_session_factory,
    custom_read_parameters,
    output_type,
    bound_load_dataframes,
):
    if output_type != "dataframe":
        pytest.skip()
    result = bound_load_dataframes(
        dataset_uuid=dataset_partition_keys.uuid,
        store=store_session_factory,
        predicates=[[("P", "==", -42)]],
        tables=["core"],
        columns={"core": ["P", "L", "TARGET"]},
        **custom_read_parameters,
    )
    assert len(result) == 0


def _gen_partition(b_c):
    b, c = b_c
    df = pd.DataFrame({"a": [1], "b": [b], "c": c})
    return {"data": [("data", df)]}


def test_read_dataset_as_dataframes_concat_primary(
    store_factory,
    custom_read_parameters,
    bound_load_dataframes,
    output_type,
    metadata_version,
):
    if output_type != "dataframe":
        pytest.skip()
    partitions = []
    for part_info in [["1", "H"], ["1", "G"], ["2", "H"], ["2", "G"]]:
        partitions.append(_gen_partition(part_info))

    store_dataframes_as_dataset(
        dfs=partitions,
        store=store_factory,
        dataset_uuid="partitioned_uuid",
        metadata_version=metadata_version,
        partition_on=["a", "b"],
    )

    result = bound_load_dataframes(
        dataset_uuid="partitioned_uuid",
        store=store_factory,
        concat_partitions_on_primary_index=True,
        predicates=[[("b", "==", "1")]],
        **custom_read_parameters,
    )
    result_df = result[0]["data"].sort_values(by="c")

    expected_df = pd.DataFrame(
        {"a": [1, 1], "b": ["1", "1"], "c": ["G", "H"]}
    ).sort_values(by="c")
    # Concatenated DataFrames have also a concatenated index.
    # Reflect this in the test.
    expected_df.index = [0, 0]

    pdt.assert_frame_equal(expected_df, result_df, check_like=True)


@pytest.mark.parametrize("dispatch_by", ["A", "B", "C"])
def test_read_dataset_as_dataframes_dispatch_by_single_col(
    store_factory,
    bound_load_dataframes,
    backend_identifier,
    dispatch_by,
    output_type,
    metadata_version,
):
    if output_type == "table":
        pytest.skip()
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
    partitions = [{"data": [("data", c)]} for c in clusters]

    store_dataframes_as_dataset__iter(
        df_generator=partitions,
        store=store_factory,
        dataset_uuid="partitioned_uuid",
        metadata_version=metadata_version,
        partition_on=["A", "B"],
        secondary_indices=["C"],
    )

    # Dispatch by primary index "A"
    dispatched_a = bound_load_dataframes(
        dataset_uuid="partitioned_uuid", store=store_factory, dispatch_by=[dispatch_by]
    )

    unique_a = set()
    for part in dispatched_a:
        if isinstance(part, MetaPartition):
            data = part.data["data"]
        else:
            data = part["data"]
        unique_dispatch = data[dispatch_by].unique()
        assert len(unique_dispatch) == 1
        unique_dispatch[0] not in unique_a
        unique_a.add(unique_dispatch[0])


def test_read_dataset_as_dataframes_dispatch_by_multi_col(
    store_factory,
    bound_load_dataframes,
    backend_identifier,
    output_type,
    metadata_version,
):
    if output_type == "table":
        pytest.skip()
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
    partitions = [{"data": [("data", c)]} for c in clusters]

    store_dataframes_as_dataset__iter(
        df_generator=partitions,
        store=store_factory,
        dataset_uuid="partitioned_uuid",
        metadata_version=metadata_version,
        partition_on=["A", "B"],
        secondary_indices=["C"],
    )
    for dispatch_by in permutations(("A", "B", "C"), 2):
        dispatched = bound_load_dataframes(
            dataset_uuid="partitioned_uuid",
            store=store_factory,
            dispatch_by=dispatch_by,
        )
        uniques = pd.DataFrame(columns=dispatch_by)
        for part in dispatched:
            if isinstance(part, MetaPartition):
                data = part.data["data"]
            else:
                data = part["data"]
            unique_dispatch = data[list(dispatch_by)].drop_duplicates()
            assert len(unique_dispatch) == 1
            row = unique_dispatch
            uniques.append(row)
        assert not any(uniques.duplicated())


def test_read_dataset_as_dataframes(
    dataset,
    store_session_factory,
    dataset_factory,
    use_dataset_factory,
    bound_load_dataframes,
    use_categoricals,
    output_type,
    label_filter,
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
        label_filter=label_filter,
        dates_as_object=dates_as_object,
    )


def test_load_dataset_metadata(
    dataset, store_session_factory, bound_load_metapartitions
):
    result = bound_load_metapartitions(
        dataset_uuid=dataset.uuid,
        store=store_session_factory,
        load_dataset_metadata=True,
    )
    for mp in result:
        assert set(mp.dataset_metadata.keys()) == {"creation_time", "dataset"}


def test_read_dataset_as_dataframes_columns_projection(
    store_factory, bound_load_dataframes, metadata_version
):
    table_name = "core"

    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return {"label": str(c), "data": [(table_name, df)]}

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
        dataset_uuid=dataset_uuid,
        store=store_factory,
        columns={table_name: ["a", "b", "c"]},
    )
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data[table_name] for mp in result]
    elif isinstance(probe, dict):
        result_dfs = [mp[table_name] for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)

    expected_df = pd.DataFrame({"a": [1, 1], "b": [1, 1], "c": [100, 100]})
    pdt.assert_frame_equal(expected_df, result_df, check_like=True)


def test_read_dataset_as_dataframes_columns_primary_index_only(
    store_factory, bound_load_dataframes, metadata_version
):
    table_name = "core"

    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return {"label": str(c), "data": [(table_name, df)]}

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
        dataset_uuid=dataset_uuid, store=store_factory, columns={table_name: ["a", "b"]}
    )
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data[table_name] for mp in result]
    elif isinstance(probe, dict):
        result_dfs = [mp[table_name] for mp in result]
    else:
        result_dfs = result
    result_df = pd.concat(result_dfs).reset_index(drop=True)

    expected_df = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
    pdt.assert_frame_equal(expected_df, result_df, check_like=True)


def test_empty_predicate_pushdown_empty_col_projection(
    dataset, store_session_factory, bound_load_dataframes, backend_identifier
):
    table_name = "core"
    result = bound_load_dataframes(
        dataset_uuid=dataset.uuid,
        tables=table_name,
        store=store_session_factory,
        columns={table_name: []},
        predicates=[[("P", "==", 12345678)]],  # this product doesn't exist
    )

    if backend_identifier.startswith("dask"):
        pytest.xfail("Output of dask for empty results is currently inconsistent")
    probe = result[0]

    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data[table_name] for mp in result]
    elif isinstance(probe, dict):
        result_dfs = [mp[table_name] for mp in result]
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
    custom_read_parameters,
    output_type,
    partition_on,
    datetype,
    comp,
):
    table_name = "core"

    def _f(b_c):
        b, c = b_c
        df = pd.DataFrame({"a": [1, 1], "b": [b, b], "c": c, "d": [b, b + 1]})
        return {"label": gen_uuid(), "data": [(table_name, df)]}

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
        tables=table_name,
        store=store_factory,
        predicates=[[("c", comp, datetype(2000, 1, 1))]],
        dates_as_object=True,
        **custom_read_parameters,
    )
    if output_type != "dataframe":
        return

    assert len(result) == 1
    dct = result[0]
    assert set(dct.keys()) == {table_name}
    df_actual = dct[table_name]

    df_expected = in_partitions[0]["data"][0][1]
    pdt.assert_frame_equal(df_actual, df_expected, check_like=True)


def test_binary_column_metadata(store_factory, bound_load_dataframes):
    table_name = "core"
    df = {
        "label": "part1",
        "data": [(table_name, pd.DataFrame({b"int_col": [1], "🙈".encode(): [2]}))],
    }

    store_dataframes_as_dataset(
        dfs=[df], store=store_factory, dataset_uuid="dataset_uuid"
    )

    result = bound_load_dataframes(
        dataset_uuid="dataset_uuid", store=store_factory, tables=table_name
    )

    probe = result[0]
    if isinstance(probe, MetaPartition):
        result_dfs = [mp.data[table_name] for mp in result]
    elif isinstance(probe, dict):
        result_dfs = [mp[table_name] for mp in result]
    else:
        result_dfs = result
    df = pd.concat(result_dfs).reset_index(drop=True)

    # Assert column names are of type `str`, instead of `bytes` objects
    assert set(df.columns.map(type)) == {str}
