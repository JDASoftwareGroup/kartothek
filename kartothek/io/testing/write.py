# -*- coding: utf-8 -*-
# pylint: disable=E1101


import string
from collections import OrderedDict

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.metapartition import MetaPartition
from kartothek.serialization import DataFrameSerializer


def test_file_structure_dataset_v4(store_factory, bound_store_dataframes):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_helper = pd.DataFrame(
        {"P": np.arange(0, 10), "info": string.ascii_lowercase[:10]}
    )

    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
        {
            "label": "cluster_2",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
    ]

    dataset = bound_store_dataframes(
        df_list, store=store_factory, dataset_uuid="dataset_uuid", metadata_version=4
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    store = store_factory()
    # TODO: json -> msgpack
    expected_keys = set(
        [
            "dataset_uuid.by-dataset-metadata.json",
            "dataset_uuid/helper/cluster_1.parquet",
            "dataset_uuid/helper/cluster_2.parquet",
            "dataset_uuid/helper/_common_metadata",
            "dataset_uuid/core/cluster_1.parquet",
            "dataset_uuid/core/cluster_2.parquet",
            "dataset_uuid/core/_common_metadata",
        ]
    )
    assert set(expected_keys) == set(store.keys())


def test_file_structure_dataset_v4_partition_on(store_factory, bound_store_dataframes):
    store = store_factory()
    assert set(store.keys()) == set()
    df = pd.DataFrame(
        {"P": [1, 2, 3, 1, 2, 3], "L": [1, 1, 1, 2, 2, 2], "TARGET": np.arange(10, 16)}
    )
    df_helper = pd.DataFrame(
        {
            "P": [1, 2, 3, 1, 2, 3],
            "L": [1, 1, 1, 2, 2, 2],
            "info": string.ascii_lowercase[:2],
        }
    )

    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
        {
            "label": "cluster_2",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
    ]
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

    expected_keys = set(
        [
            "dataset_uuid.by-dataset-metadata.json",
            "dataset_uuid/helper/P=1/L=1/cluster_1.parquet",
            "dataset_uuid/helper/P=1/L=1/cluster_2.parquet",
            "dataset_uuid/helper/P=1/L=2/cluster_1.parquet",
            "dataset_uuid/helper/P=1/L=2/cluster_2.parquet",
            "dataset_uuid/helper/P=2/L=1/cluster_1.parquet",
            "dataset_uuid/helper/P=2/L=1/cluster_2.parquet",
            "dataset_uuid/helper/P=2/L=2/cluster_1.parquet",
            "dataset_uuid/helper/P=2/L=2/cluster_2.parquet",
            "dataset_uuid/helper/P=3/L=1/cluster_1.parquet",
            "dataset_uuid/helper/P=3/L=1/cluster_2.parquet",
            "dataset_uuid/helper/P=3/L=2/cluster_1.parquet",
            "dataset_uuid/helper/P=3/L=2/cluster_2.parquet",
            "dataset_uuid/helper/_common_metadata",
            "dataset_uuid/core/P=1/L=1/cluster_1.parquet",
            "dataset_uuid/core/P=1/L=1/cluster_2.parquet",
            "dataset_uuid/core/P=1/L=2/cluster_1.parquet",
            "dataset_uuid/core/P=1/L=2/cluster_2.parquet",
            "dataset_uuid/core/P=2/L=1/cluster_1.parquet",
            "dataset_uuid/core/P=2/L=1/cluster_2.parquet",
            "dataset_uuid/core/P=2/L=2/cluster_1.parquet",
            "dataset_uuid/core/P=2/L=2/cluster_2.parquet",
            "dataset_uuid/core/P=3/L=1/cluster_1.parquet",
            "dataset_uuid/core/P=3/L=1/cluster_2.parquet",
            "dataset_uuid/core/P=3/L=2/cluster_1.parquet",
            "dataset_uuid/core/P=3/L=2/cluster_2.parquet",
            "dataset_uuid/core/_common_metadata",
        ]
    )

    assert set(expected_keys) == set(store.keys())


def test_file_structure_dataset_v4_partition_on_second_table_no_index_col(
    store_factory, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 2), "L": np.arange(0, 2), "TARGET": np.arange(10, 12)}
    )
    df_helper = pd.DataFrame({"P": [0, 0, 1], "info": string.ascii_lowercase[:2]})

    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
        {
            "label": "cluster_2",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
    ]

    with pytest.raises(Exception):
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P", "L"],
            metadata_version=4,
        )


def test_file_structure_dataset_v4_partition_on_second_table_no_index_col_simple_group(
    store_factory, bound_store_dataframes
):
    """
    Pandas seems to stop evaluating the groupby expression if the dataframes after the first column split
    is of length 1. This seems to be an optimization which should, however, still raise a KeyError
    """
    df = pd.DataFrame(
        {"P": np.arange(0, 2), "L": np.arange(0, 2), "TARGET": np.arange(10, 12)}
    )
    df_helper = pd.DataFrame({"P": [0, 1], "info": string.ascii_lowercase[:2]})

    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
        {
            "label": "cluster_2",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
    ]

    with pytest.raises(Exception):
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P", "L"],
            metadata_version=4,
        )


def test_store_dataframes_as_dataset(
    store_factory, metadata_version, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_helper = pd.DataFrame(
        {"P": np.arange(0, 10), "info": string.ascii_lowercase[:10]}
    )

    df_list = [
        {
            "label": "cluster_1",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
        {
            "label": "cluster_2",
            "data": [("core", df.copy(deep=True)), ("helper", df_helper)],
        },
    ]

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
    assert any([sorted(p) == ["cluster_1", "cluster_2"] for p in index_dct.values()])

    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_1"].files["core"], store=store
    )
    pdt.assert_frame_equal(df, df_stored)
    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_2"].files["core"], store=store
    )
    pdt.assert_frame_equal(df, df_stored)
    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_1"].files["helper"], store=store
    )
    pdt.assert_frame_equal(df_helper, df_stored)
    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_2"].files["helper"], store=store
    )
    pdt.assert_frame_equal(df_helper, df_stored)


def test_store_dataframes_as_dataset_empty_dataframe(
    store_factory, metadata_version, df_all_types, bound_store_dataframes
):
    """
    Test that writing an empty column succeeds.
    In particular, this may fail due to too strict schema validation.
    """
    df_empty = df_all_types.drop(0)

    # Store a second table with shared columns. All shared columns must be of the same type
    # This may fail in the presence of empty partitions if the schema validation doesn't account for it
    df_shared_cols = df_all_types.loc[:, df_all_types.columns[:3]]
    df_shared_cols["different_col"] = "a"

    assert df_empty.empty
    df_list = [
        {
            "label": "cluster_1",
            "data": [("tableA", df_empty), ("tableB", df_shared_cols.copy(deep=True))],
        },
        {
            "label": "cluster_2",
            "data": [
                ("tableA", df_all_types),
                ("tableB", df_shared_cols.copy(deep=True)),
            ],
        },
    ]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    store = store_factory()
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions

    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_1"].files["tableA"], store=store
    )
    pdt.assert_frame_equal(df_empty, df_stored)

    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_2"].files["tableA"], store=store
    )
    # Roundtrips for type date are not type preserving
    df_stored["date"] = df_stored["date"].dt.date
    pdt.assert_frame_equal(df_all_types, df_stored)

    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_1"].files["tableB"], store=store
    )
    pdt.assert_frame_equal(df_shared_cols, df_stored)
    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_2"].files["tableB"], store=store
    )
    pdt.assert_frame_equal(df_shared_cols, df_stored)


def test_store_dataframes_as_dataset_batch_mode(
    store_factory, metadata_version, bound_store_dataframes
):
    values_p1 = [1, 2, 3]
    values_p2 = [4, 5, 6]
    df = pd.DataFrame({"P": values_p1})
    df2 = pd.DataFrame({"P": values_p2})

    df_list = [
        [
            {
                "label": "cluster_1",
                "data": [("core", df)],
                "indices": {
                    "P": ExplicitSecondaryIndex(
                        "P", {v: ["cluster_1"] for v in values_p1}
                    )
                },
            },
            {
                "label": "cluster_2",
                "data": [("core", df2)],
                "indices": {
                    "P": ExplicitSecondaryIndex(
                        "P", {v: ["cluster_2"] for v in values_p2}
                    )
                },
            },
        ]
    ]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
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

    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_1"].files["core"], store=store
    )
    pdt.assert_frame_equal(df, df_stored)
    df_stored = DataFrameSerializer.restore_dataframe(
        key=dataset.partitions["cluster_2"].files["core"], store=store
    )
    pdt.assert_frame_equal(df2, df_stored)

    assert stored_dataset.indices["P"].to_dict() == {
        1: np.array(["cluster_1"], dtype=object),
        2: np.array(["cluster_1"], dtype=object),
        3: np.array(["cluster_1"], dtype=object),
        4: np.array(["cluster_2"], dtype=object),
        5: np.array(["cluster_2"], dtype=object),
        6: np.array(["cluster_2"], dtype=object),
    }


def test_store_dataframes_as_dataset_auto_uuid(
    store_factory, metadata_version, mock_uuid, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_helper = pd.DataFrame(
        {"P": np.arange(0, 10), "info": string.ascii_lowercase[:10]}
    )

    df_list = [
        {
            "label": "cluster_1",
            "data": [
                ("core", df.copy(deep=True)),
                ("helper", df_helper.copy(deep=True)),
            ],
        },
        {
            "label": "cluster_2",
            "data": [
                ("core", df.copy(deep=True)),
                ("helper", df_helper.copy(deep=True)),
            ],
        },
    ]

    dataset = bound_store_dataframes(
        df_list, store=store_factory, metadata_version=metadata_version
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2

    stored_dataset = DatasetMetadata.load_from_store(
        "auto_dataset_uuid", store_factory()
    )
    assert dataset.uuid == stored_dataset.uuid
    assert dataset.metadata == stored_dataset.metadata
    assert dataset.partitions == stored_dataset.partitions


def test_store_dataframes_as_dataset_list_input(
    store_factory, metadata_version, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )
    df2 = pd.DataFrame(
        {
            "P": np.arange(100, 110),
            "L": np.arange(100, 110),
            "TARGET": np.arange(10, 20),
        }
    )
    df_list = [df, df2]

    dataset = bound_store_dataframes(
        df_list,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    assert isinstance(dataset, DatasetMetadata)
    assert len(dataset.partitions) == 2
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store_factory())
    assert dataset == stored_dataset


def test_store_dataframes_as_dataset_mp_partition_on_none(
    metadata_version, store, store_factory, bound_store_dataframes
):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df2 = pd.DataFrame({"P": np.arange(0, 10), "info": np.arange(100, 110)})

    mp = MetaPartition(
        label=gen_uuid(),
        data={"core": df, "helper": df2},
        metadata_version=metadata_version,
    )

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
    input_ = [
        {
            "label": "label",
            "data": [("order_proposals", df.head(0))],
            "indices": {"location": {}},
        },
        {
            "label": "label",
            "data": [("order_proposals", df)],
            "indices": {"location": {k: ["label"] for k in df["location"].unique()}},
        },
    ]
    dataset = bound_store_dataframes(
        input_,
        store=store_factory,
        dataset_uuid="dataset_uuid",
        metadata_version=4,
        partition_on=["other"],
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
    df_list = [{"label": "cluster_1", "data": [("core", df)]} for df in dfs]

    if ok:
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P"],
            metadata_version=4,
        )
    else:
        with pytest.raises(Exception) as exc:
            bound_store_dataframes(
                df_list,
                store=store_factory,
                dataset_uuid="dataset_uuid",
                partition_on=["P"],
                metadata_version=4,
            )
        assert (
            "Schemas for table 'core' of dataset 'dataset_uuid' are not compatible!"
            in _exception_str(exc.value)
        )


def test_schema_check_write_shared(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame(
        {"P": pd.Series([1], dtype=np.int64), "X": pd.Series([1], dtype=np.int64)}
    )
    df2 = pd.DataFrame(
        {"P": pd.Series([1], dtype=np.uint64), "Y": pd.Series([1], dtype=np.int64)}
    )
    df_list = [
        {"label": "cluster_1", "data": [("core", df1)]},
        {"label": "cluster_2", "data": [("prediction", df2)]},
    ]
    with pytest.raises(Exception) as exc:
        bound_store_dataframes(
            df_list,
            store=store_factory,
            dataset_uuid="dataset_uuid",
            partition_on=["P"],
            metadata_version=4,
        )
    assert 'Found incompatible entries for column "P"' in str(exc.value)


def test_schema_check_write_nice_error(store_factory, bound_store_dataframes):
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
        {"label": "uuid1", "data": [("core", df1)]},
        {"label": "uuid2", "data": [("core", df2)]},
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
        """Schemas for table 'core' of dataset 'dataset_uuid' are not compatible!

Schema violation

Origin schema: {core/P=2/Q=2/uuid2}
Origin reference: {core/P=1/Q=2/uuid1}

Diff:
"""
    )


def test_schema_check_write_cut_error(store_factory, bound_store_dataframes):
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
        {"label": "uuid1", "data": [("core", df1)]},
        {"label": "uuid2", "data": [("core", df2)]},
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
        """Schemas for table 'core' of dataset 'dataset_uuid' are not compatible!

Schema violation

Origin schema: {core/P=2/Q=99/uuid2}
Origin reference: {core/P=1/Q=99/uuid1}

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

    df_list = [
        {"label": "cluster_1", "data": [("core", df)]},
        {"label": "cluster_2", "data": [("core", df_2)]},
    ]

    # Also test `df_list` in reverse order, as this could lead to different results
    for dfs in [df_list, list(reversed(df_list))]:
        with pytest.raises(
            Exception, match=r"Schemas for table .* of dataset .* are not compatible!"
        ):
            return bound_store_dataframes(
                dfs, store=store_factory, metadata_version=metadata_version
            )


def test_table_consistency_resistance(
    store_factory, metadata_version, bound_store_dataframes
):
    df = pd.DataFrame({"P": np.arange(0, 10)})

    df_helper = pd.DataFrame(
        {"P": np.arange(15, 35), "info": string.ascii_lowercase[:10]}
    )

    df_list = [
        {"label": "cluster_1", "data": [("core", df)]},
        {"label": "cluster_2", "data": [("core", df), ("helper", df_helper)]},
    ]

    store_kwargs = dict(store=store_factory, metadata_version=metadata_version)
    metadata1 = bound_store_dataframes(df_list, **store_kwargs)

    metadata2 = bound_store_dataframes(list(reversed(df_list)), **store_kwargs)

    assert set(metadata1.tables) == set(metadata2.tables) == {"core", "helper"}


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


def test_store_empty_dataframes_partition_on(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame({"x": [1], "y": [1]}).iloc[[]]
    md1 = bound_store_dataframes(
        [df1], store=store_factory, dataset_uuid="uuid", partition_on=["x"]
    )
    assert md1.tables == ["table"]
    assert set(md1.table_meta["table"].names) == set(df1.columns)

    df2 = pd.DataFrame({"x": [1], "y": [1], "z": [1]}).iloc[[]]
    md2 = bound_store_dataframes(
        [df2],
        store=store_factory,
        dataset_uuid="uuid",
        partition_on=["x"],
        overwrite=True,
    )
    assert md2.tables == ["table"]
    assert set(md2.table_meta["table"].names) == set(df2.columns)

    df3 = pd.DataFrame({"x": [1], "y": [1], "a": [1]}).iloc[[]]
    md3 = bound_store_dataframes(
        [{"table2": df3}],
        store=store_factory,
        dataset_uuid="uuid",
        partition_on=["x"],
        overwrite=True,
    )
    assert md3.tables == ["table2"]
    assert set(md3.table_meta["table2"].names) == set(df3.columns)


def test_store_overwrite_none(store_factory, bound_store_dataframes):
    df1 = pd.DataFrame({"x": [1], "y": [1]})
    md1 = bound_store_dataframes(
        [df1], store=store_factory, dataset_uuid="uuid", partition_on=["x"]
    )
    assert md1.tables == ["table"]
    assert set(md1.table_meta["table"].names) == set(df1.columns)

    md2 = bound_store_dataframes(
        [{}],
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
