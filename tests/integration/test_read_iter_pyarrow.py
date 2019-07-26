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

import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import store_dataframes_as_dataset
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


def _gen_partition(b_c):
    b, c = b_c
    df = pd.DataFrame({"a": [1], "b": [b], "c": c})
    return {"data": [("data", df)]}


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
