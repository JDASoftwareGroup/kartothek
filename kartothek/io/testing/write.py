# -*- coding: utf-8 -*-
# pylint: disable=E1101


from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from storefact import get_store_from_url

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import read_table
from kartothek.io_components.metapartition import MetaPartition
from kartothek.serialization import DataFrameSerializer


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


def test_store_input_types(store_input_types, bound_store_dataframes):
    from kartothek.serialization.testing import get_dataframe_not_nested

    dataset_uuid = "dataset_uuid"
    df = get_dataframe_not_nested(10)

    assert bound_store_dataframes(
        [df],
        dataset_uuid=dataset_uuid,
        store=store_input_types,
        partition_on=[df.columns[0]],
        secondary_indices=[df.columns[1]],
    )


def test_file_structure_dataset_v4(store_factory, bound_store_dataframes):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df.copy(deep=True), df.copy(deep=True)]

    dataset = bound_store_dataframes(
        df_list, store=store_factory, dataset_uuid="dataset_uuid", metadata_version=4
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    store = store_factory()

    assert len(store.keys()) == 4
    assert "dataset_uuid/table/_common_metadata" in store
    assert "dataset_uuid.by-dataset-metadata.json" in store


def test_file_structure_dataset_v4_partition_on(store_factory, bound_store_dataframes):
    store = store_factory()
    assert set(store.keys()) == set()
    df = pd.DataFrame(
        {"P": [1, 2, 3, 1, 2, 3], "L": [1, 1, 1, 2, 2, 2], "TARGET": np.arange(10, 16)}
    )

    df_list = [df.copy(deep=True), df.copy(deep=True)]
    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        partition_on=["P", "L"],
        metadata_version=4,
    )

    assert isinstance(dataset, DatasetMetadata)

    assert dataset.partition_keys == ["P", "L"]

    assert len(dataset.partitions) == 12

    store = store_factory()
    actual_keys = set(store.keys())
    assert len(actual_keys) == 14  # one per partition + json + schema


def test_store_dataframes_as_dataset(
    store_factory, metadata_version, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df.copy(deep=True), df.copy(deep=True)]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
        secondary_indices=["P"],
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    assert "P" in dataset.indices

    store = store_factory()
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions

    index_dct = stored_dataset.indices["P"].load(store).index_dct
    assert sorted(index_dct.keys()) == list(range(0, 10))

    counter = 0
    for k in store.keys():
        if "parquet" in k and "indices" not in k:
            counter += 1
            df_stored = DataFrameSerializer.restore_dataframe(key=k, store=store)
            pdt.assert_frame_equal(df, df_stored)
    assert counter == 2


def test_store_dataframes_as_dataset_empty_dataframe(
    store_factory, metadata_version, df_all_types, bound_store_dataframes
):
    """
    Test that writing an empty column succeeds.
    In particular, this may fail due to too strict schema validation.
    """
    df_empty = df_all_types.drop(0)

    assert df_empty.empty
    df_list = [df_empty]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 1

    store = store_factory()
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions

    df_stored = DataFrameSerializer.restore_dataframe(
        key=next(iter(dataset.partitions.values())).files["table"], store=store
    )
    pdt.assert_frame_equal(df_empty, df_stored)


def test_store_dataframes_as_dataset_batch_mode(
    store_factory, metadata_version, bound_store_dataframes
):
    # TODO: Kick this out?
    values_p1 = [1, 2, 3]
    values_p2 = [4, 5, 6]
    df = pd.DataFrame({"P": values_p1})
    df2 = pd.DataFrame({"P": values_p2})

    df_list = [[df, df2]]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
        secondary_indices="P",
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    store = store_factory()
    stored_dataset = DatasetMetadata.load_from_store(
        "dataset_uuid", store
    ).load_all_indices(store)
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions

    assert "P" in dataset.indices


def test_store_dataframes_as_dataset_auto_uuid(
    store_factory, metadata_version, mock_uuid, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df.copy(deep=True)]

    dataset = bound_store_dataframes(
        df_list, store=store_factory, metadata_version=metadata_version
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 1

    stored_dataset = DatasetMetadata.load_from_store(
        "auto_dataset_uuid", store_factory()
    )
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions


def test_store_dataframes_as_dataset_mp_partition_on_none(
    metadata_version, store, store_factory, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    mp = MetaPartition(label=gen_uuid(), data=df, metadata_version=metadata_version)

    df_list = [None, mp]
    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
        partition_on=["P"],
    )

    assert isinstance(dataset, DatasetMetadata)
    assert dataset.partition_keys == ["P"]
    assert len(dataset.partitions) == 10
    assert dataset.metadata_version == metadata_version

    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)

    assert dataset == stored_dataset


def test_store_dataframes_partition_on(store_factory, bound_store_dataframes):
    df = pd.DataFrame(
        OrderedDict([("location", ["0", "1", "2"]), ("other", ["a", "a", "a"])])
    )

    # First partition is empty, test this edgecase
    input_ = [df.head(0), df]
    dataset = bound_store_dataframes(
        input_,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=4,
        partition_on=["other"],
        secondary_indices="location",
    )

    assert len(dataset.partitions) == 1
    assert len(dataset.indices) == 1

    assert dataset.partition_keys == ["other"]


def _exception_str(exception):
    """
    Extract the exception message, even if this is a re-throw of an exception
    in distributed.
    """
    if isinstance(exception, ValueError) and exception.args[0] == "Long error message":
        return exception.args[1]
    return str(exception)


@pytest.mark.parametrize(
    "dfs,ok",
    [
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int32),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int16),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int16),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int32),
                        "X": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            True,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.uint64),
                    }
                ),
            ],
            False,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1], dtype=np.int64),
                        "X": pd.Series([1], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([2], dtype=np.int64),
                        "X": pd.Series([2], dtype=np.int64),
                        "Y": pd.Series([2], dtype=np.int64),
                    }
                ),
            ],
            False,
        ),
        (
            [
                pd.DataFrame(
                    {
                        "P": pd.Series([1, 2], dtype=np.int64),
                        "X": pd.Series([1, 2], dtype=np.int64),
                    }
                ),
                pd.DataFrame(
                    {
                        "P": pd.Series([3], dtype=np.int64),
                        "X": pd.Series([3], dtype=np.uint64),
                    }
                ),
            ],
            False,
        ),
    ],
)
def test_schema_check_write(dfs, ok, store_factory, bound_store_dataframes):

    if ok:
        bound_store_dataframes(
            dfs,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P"],
            metadata_version=4,
        )
    else:
        with pytest.raises(Exception) as exc:
            bound_store_dataframes(
                dfs,
                store=store_factory,
                dataset_uuid="dataset_uuid",
                partition_on=["P"],
                metadata_version=4,
            )
        assert (
            "Schemas for dataset 'dataset_uuid' are not compatible!"
            in _exception_str(exc.value)
        )


@pytest.mark.xfail(reason="mocking doesn't work for dask atm")
def test_schema_check_write_nice_error(
    store_factory, bound_store_dataframes, mock_uuid
):
    df1 = pd.DataFrame(
        {
            "P": pd.Series([1, 1], dtype=np.int64),
            "Q": pd.Series([1, 2], dtype=np.int64),
            "X": pd.Series([1, 1], dtype=np.int64),
        }
    )
    df2 = pd.DataFrame(
        {
            "P": pd.Series([2, 2], dtype=np.uint64),
            "Q": pd.Series([1, 2], dtype=np.int64),
            "X": pd.Series([1, 1], dtype=np.int64),
        }
    )
    df_list = [
        df1,
        df2,
    ]
    with pytest.raises(Exception) as exc:
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P", "Q"],
            metadata_version=4,
        )

    assert _exception_str(exc.value).startswith(
        """Schemas for dataset 'dataset_uuid' are not compatible!

Schema violation

Origin schema: {P=2/Q=2/auto_dataset_uuid}
Origin reference: {P=1/Q=2/auto_dataset_uuid}

Diff:
"""
    )


@pytest.mark.xfail(reason="mocking doesn't work for dask atm")
def test_schema_check_write_cut_error(store_factory, bound_store_dataframes, mock_uuid):
    df1 = pd.DataFrame(
        {
            "P": pd.Series([1] * 100, dtype=np.int64),
            "Q": pd.Series(range(100), dtype=np.int64),
            "X": pd.Series([1] * 100, dtype=np.int64),
        }
    )
    df2 = pd.DataFrame(
        {
            "P": pd.Series([2] * 100, dtype=np.uint64),
            "Q": pd.Series(range(100), dtype=np.int64),
            "X": pd.Series([1] * 100, dtype=np.int64),
        }
    )
    df_list = [
        df1,
        df2,
    ]
    with pytest.raises(Exception) as exc:
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P", "Q"],
            metadata_version=4,
        )
    assert _exception_str(exc.value).startswith(
        """Schemas for dataset 'dataset_uuid' are not compatible!

Schema violation

Origin schema: {P=2/Q=99/auto_dataset_uuid}
Origin reference: {P=1/Q=99/auto_dataset_uuid}

Diff:
"""
    )


def test_metadata_consistency_errors_fails(
    store_factory, metadata_version, bound_store_dataframes
):
    df = pd.DataFrame({"W": np.arange(0, 10), "L": np.arange(0, 10)})

    df_2 = pd.DataFrame(
        {"P": np.arange(10, 20), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df, df_2]

    # Also test `df_list` in reverse order, as this could lead to different results
    for dfs in [df_list, list(reversed(df_list))]:
        with pytest.raises(
            Exception, match=r"Schemas for dataset .* are not compatible!"
        ):
            return bound_store_dataframes(
                dfs, store=store_factory, metadata_version=metadata_version
            )


def test_store_dataframes_as_dataset_overwrite(
    store_factory, dataset_function, bound_store_dataframes
):
    with pytest.raises(RuntimeError):
        bound_store_dataframes(
            [pd.DataFrame()], store=store_factory, dataset_uuid=dataset_function.uuid
        )
    bound_store_dataframes(
        [pd.DataFrame()],
        store=store_factory,
        dataset_uuid=dataset_function.uuid,
        overwrite=True,
    )
    bound_store_dataframes(
        [pd.DataFrame()], store=store_factory, dataset_uuid="new_dataset_uuid"
    )


@pytest.mark.skip("What is the intended behaviour for this?")
def test_store_empty_dataframes_partition_on(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame({"x": [1], "y": [1]}).iloc[[]]
    md1 = bound_store_dataframes(
        [df1], store=store_factory, dataset_uuid="uuid", partition_on=["x"]
    )
    assert md1.tables == ["table"]
    assert set(md1.schema.names) == set(df1.columns)

    df2 = pd.DataFrame({"x": [1], "y": [1], "z": [1]}).iloc[[]]
    md2 = bound_store_dataframes(
        [df2],
        store=store_factory,
        dataset_uuid="uuid",
        partition_on=["x"],
        overwrite=True,
    )
    assert md2.tables == ["table"]
    assert set(md2.schema.names) == set(df2.columns)

    df3 = pd.DataFrame({"x": [1], "y": [1], "a": [1]}).iloc[[]]
    md3 = bound_store_dataframes(
        [{"table2": df3}],
        store=store_factory,
        dataset_uuid="uuid",
        partition_on=["x"],
        overwrite=True,
    )
    assert md3.tables == ["table2"]
    assert set(md3.schema.names) == set(df3.columns)


@pytest.mark.skip("What is the intended behaviour for this?")
def test_store_overwrite_none(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame({"x": [1], "y": [1]})
    md1 = bound_store_dataframes(
        [df1], store=store_factory, dataset_uuid="uuid", partition_on=["x"]
    )
    assert md1.tables == ["table"]
    assert set(md1.schema.names) == set(df1.columns)

    md2 = bound_store_dataframes(
        [None],
        store=store_factory,
        dataset_uuid="uuid",
        partition_on=["x"],
        overwrite=True,
    )
    assert md2.tables == []


def test_secondary_index_on_partition_column(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame({"x": [1], "y": [1]})
    with pytest.raises(
        RuntimeError, match="Cannot create secondary index on partition columns: {'x'}"
    ):
        bound_store_dataframes(
            [df1], store=store_factory, partition_on=["x"], secondary_indices=["x"]
        )


def test_non_default_table_name_roundtrip(store_factory, bound_store_dataframes):
    df = pd.DataFrame({"A": [1]})
    bound_store_dataframes(
        [df], store=store_factory, dataset_uuid="dataset_uuid", table_name="foo"
    )
    for k in store_factory():
        if k.endswith(".parquet") and "indices" not in k:
            assert "foo" in k
    result = read_table(dataset_uuid="dataset_uuid", store=store_factory)

    pdt.assert_frame_equal(df, result)
