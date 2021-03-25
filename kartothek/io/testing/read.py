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

The following fixtures should be present (see tests.read.conftest)
* ``use_categoricals`` - Whether or not the call retrievs categorical data.
* ``dates_as_object`` - Whether or not the call retrievs date columns as objects.

"""

import datetime
from functools import partial

import pandas as pd
import pandas.testing as pdt
import pytest
from storefact import get_store_from_url

from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io.iter import store_dataframes_as_dataset__iter
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition


@pytest.fixture(params=[True, False], ids=["use_categoricals", "no_categoricals"])
def use_categoricals(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["dates_as_object", "datest_as_datetime"])
def dates_as_object(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["use_factory", "no_factory"])
def use_dataset_factory(request, dates_as_object):
    return request.param


class NoPickle:
    def __getstate__(self):
        raise RuntimeError("do NOT pickle this object!")


def mark_nopickle(obj):
    setattr(obj, "_nopickle", NoPickle())


def no_pickle_store(url):
    store = get_store_from_url(url)
    mark_nopickle(store)
    return store


def no_pickle_factory(url):

    return partial(no_pickle_store, url)


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


def _gen_partition(b_c):
    b, c = b_c
    return pd.DataFrame({"a": [1], "b": [b], "c": c})


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
