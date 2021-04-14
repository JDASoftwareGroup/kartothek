from collections import OrderedDict
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.common_metadata import make_meta, store_schema_metadata
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.naming import DEFAULT_METADATA_VERSION
from kartothek.core.testing import get_dataframe_not_nested
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    _unique_label,
    parse_input_to_metapartition,
    partition_labels_from_mps,
)
from kartothek.serialization import DataFrameSerializer, ParquetSerializer


def test_store_single_dataframe_as_partition(store, metadata_version):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )
    mp = MetaPartition(label="test_label", data=df, metadata_version=metadata_version)

    meta_partition = mp.store_dataframes(
        store=store, df_serializer=ParquetSerializer(), dataset_uuid="dataset_uuid"
    )

    assert meta_partition.data is None

    expected_key = "dataset_uuid/table/test_label.parquet"

    assert meta_partition.file == expected_key
    assert meta_partition.label == "test_label"

    files_in_store = list(store.keys())

    expected_num_files = 1
    assert len(files_in_store) == expected_num_files
    stored_df = DataFrameSerializer.restore_dataframe(store=store, key=expected_key)
    pdt.assert_frame_equal(df, stored_df)
    files_in_store.remove(expected_key)
    assert len(files_in_store) == expected_num_files - 1


def test_load_dataframe_logical_conjunction(store, metadata_version):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )
    mp = MetaPartition(
        label="cluster_1",
        data=df,
        metadata_version=metadata_version,
        logical_conjunction=[("P", ">", 4)],
    )
    meta_partition = mp.store_dataframes(
        store=store, df_serializer=None, dataset_uuid="dataset_uuid"
    )
    predicates = None
    loaded_mp = meta_partition.load_dataframes(store=store, predicates=predicates)
    data = pd.DataFrame(
        {"P": [5, 6, 7, 8, 9], "L": [5, 6, 7, 8, 9], "TARGET": [15, 16, 17, 18, 19]}
    ).set_index(np.arange(5, 10))
    pdt.assert_frame_equal(loaded_mp.data, data)

    predicates = [[("L", ">", 6), ("TARGET", "<", 18)]]
    loaded_mp = meta_partition.load_dataframes(store=store, predicates=predicates)
    data = pd.DataFrame({"P": [7], "L": [7], "TARGET": [17]}).set_index(np.array([7]))
    pdt.assert_frame_equal(loaded_mp.data, data)

    predicates = [[("L", ">", 2), ("TARGET", "<", 17)], [("TARGET", "==", 19)]]
    loaded_mp = meta_partition.load_dataframes(store=store, predicates=predicates)
    data = pd.DataFrame(
        {"P": [5, 6, 9], "L": [5, 6, 9], "TARGET": [15, 16, 19]}
    ).set_index(np.array([5, 6, 9]))
    pdt.assert_frame_equal(loaded_mp.data, data)


@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_load_dataframes(
    meta_partitions_files_only, store_session, predicate_pushdown_to_io
):
    expected_df = pd.DataFrame(
        OrderedDict(
            [("P", [1]), ("L", [1]), ("TARGET", [1]), ("DATE", [date(2010, 1, 1)])]
        )
    )
    mp = meta_partitions_files_only[0]
    assert mp.file
    assert mp.data is not None
    mp = meta_partitions_files_only[0].load_dataframes(
        store=store_session, predicate_pushdown_to_io=predicate_pushdown_to_io
    )
    assert mp.data is not None
    data = mp.data

    pdt.assert_frame_equal(data, expected_df, check_dtype=False)

    empty_mp = MetaPartition("empty_mp", metadata_version=mp.metadata_version)
    empty_mp.load_dataframes(
        store_session, predicate_pushdown_to_io=predicate_pushdown_to_io
    )
    assert empty_mp.data is None


def test_remove_dataframes(meta_partitions_files_only, store_session):
    mp = meta_partitions_files_only[0].load_dataframes(store=store_session)
    assert mp.data is not None
    mp = mp.remove_dataframes()
    assert mp.data is None


def test_load_dataframes_selective(meta_partitions_files_only, store_session):
    expected_df = pd.DataFrame(
        OrderedDict(
            [("P", [1]), ("L", [1]), ("TARGET", [1]), ("DATE", [date(2010, 1, 1)])]
        )
    )
    mp = meta_partitions_files_only[0]
    assert mp.file is not None
    assert mp.data is not None
    mp = meta_partitions_files_only[0].load_dataframes(store=store_session)

    data = mp.data

    pdt.assert_frame_equal(data, expected_df, check_dtype=False)


def test_load_dataframes_columns_projection(
    meta_partitions_evaluation_files_only, store_session
):
    expected_df = pd.DataFrame(OrderedDict([("P", [1]), ("L", [1]), ("HORIZON", [1])]))
    mp = meta_partitions_evaluation_files_only[0]
    assert mp.file is not None
    assert mp.data is not None
    mp = meta_partitions_evaluation_files_only[0].load_dataframes(
        store=store_session, columns=["P", "L", "HORIZON"]
    )

    data = mp.data

    pdt.assert_frame_equal(data, expected_df, check_dtype=False)


def test_load_dataframes_columns_raises_missing(
    meta_partitions_evaluation_files_only, store_session
):
    mp = meta_partitions_evaluation_files_only[0]
    assert mp.file is not None
    assert mp.data is not None
    with pytest.raises(ValueError) as e:
        meta_partitions_evaluation_files_only[0].load_dataframes(
            store=store_session, columns=["P", "L", "HORIZON", "foo", "bar"]
        )
    assert str(e.value) == "Columns cannot be found in stored dataframe: bar, foo"


def test_from_dict():
    df = pd.DataFrame({"a": [1]})
    dct = {"data": df, "label": "test_label"}
    meta_partition = MetaPartition.from_dict(dct)

    pdt.assert_frame_equal(meta_partition.data, df)
    assert meta_partition.metadata_version == DEFAULT_METADATA_VERSION


def test_eq():
    df = pd.DataFrame({"a": [1]})
    df_same = pd.DataFrame({"a": [1]})
    df_other = pd.DataFrame({"a": [2]})
    df_diff_col = pd.DataFrame({"b": [1]})
    df_diff_type = pd.DataFrame({"b": [1.0]})

    meta_partition = MetaPartition.from_dict({"label": "test_label", "data": df})
    assert meta_partition == meta_partition

    meta_partition_same = MetaPartition.from_dict(
        {"label": "test_label", "data": df_same}
    )
    assert meta_partition == meta_partition_same

    meta_partition_diff_label = MetaPartition.from_dict(
        {"label": "another_label", "data": df}
    )
    assert meta_partition != meta_partition_diff_label
    assert meta_partition_diff_label != meta_partition

    meta_partition_diff_files = MetaPartition.from_dict(
        {"label": "another_label", "data": df, "file": "something"}
    )
    assert meta_partition != meta_partition_diff_files
    assert meta_partition_diff_files != meta_partition

    meta_partition_diff_col = MetaPartition.from_dict(
        {"label": "test_label", "data": df_diff_col}
    )
    assert meta_partition != meta_partition_diff_col
    assert meta_partition_diff_col != meta_partition

    meta_partition_diff_type = MetaPartition.from_dict(
        {"label": "test_label", "data": df_diff_type}
    )
    assert meta_partition != meta_partition_diff_type
    assert meta_partition_diff_type != meta_partition

    meta_partition_diff_metadata = MetaPartition.from_dict(
        {"label": "test_label", "data": df_diff_type}
    )
    assert meta_partition != meta_partition_diff_metadata
    assert meta_partition_diff_metadata != meta_partition

    meta_partition_different_df = MetaPartition.from_dict(
        {"label": "test_label", "data": df_other}
    )
    assert not meta_partition == meta_partition_different_df

    meta_partition_empty_data = MetaPartition.from_dict(
        {"label": "test_label", "data": None}
    )
    assert meta_partition_empty_data == meta_partition_empty_data

    assert not meta_partition == "abc"


def test_add_nested_to_plain():
    mp = MetaPartition(
        label="label_1",
        file="file",
        data=pd.DataFrame({"test": [1, 2, 3]}),
        indices={"test": [1, 2, 3]},
    )

    to_nest = [
        MetaPartition(
            label="label_2",
            data=pd.DataFrame({"test": [4, 5, 6]}),
            indices={"test": [4, 5, 6]},
        ),
        MetaPartition(
            label="label_22",
            data=pd.DataFrame({"test": [4, 5, 6]}),
            indices={"test": [4, 5, 6]},
        ),
    ]
    mp_nested = to_nest[0].add_metapartition(to_nest[1])

    mp_add_nested = mp.add_metapartition(mp_nested)
    mp_iter = mp.add_metapartition(to_nest[0]).add_metapartition(to_nest[1])

    assert mp_add_nested == mp_iter


def test_add_nested_to_nested():
    mps1 = [
        MetaPartition(
            label="label_1",
            file="file",
            data=pd.DataFrame({"test": [1, 2, 3]}),
            indices={"test": [1, 2, 3]},
        ),
        MetaPartition(
            label="label_33",
            file="file",
            data=pd.DataFrame({"test": [1, 2, 3]}),
            indices={"test": [1, 2, 3]},
        ),
    ]

    mpn_1 = mps1[0].add_metapartition(mps1[1])

    mps2 = [
        MetaPartition(
            label="label_2",
            data=pd.DataFrame({"test": [4, 5, 6]}),
            indices={"test": [4, 5, 6]},
        ),
        MetaPartition(
            label="label_22",
            data=pd.DataFrame({"test": [4, 5, 6]}),
            indices={"test": [4, 5, 6]},
        ),
    ]
    mpn_2 = mps2[0].add_metapartition(mps2[1])

    mp_nested_merge = mpn_1.add_metapartition(mpn_2)

    mp_iter = mps1.pop()
    for mp_ in [*mps1, *mps2]:
        mp_iter = mp_iter.add_metapartition(mp_)

    assert mp_nested_merge == mp_iter


def test_eq_nested():
    mp_1 = MetaPartition(
        label="label_1",
        file="file",
        data=pd.DataFrame({"test": [1, 2, 3]}),
        indices={"test": [1, 2, 3]},
    )

    mp_2 = MetaPartition(
        label="label_2",
        data=pd.DataFrame({"test": [4, 5, 6]}),
        indices={"test": [4, 5, 6]},
    )

    mp = mp_1.add_metapartition(mp_2)

    assert mp == mp
    assert mp != mp_2
    assert mp_2 != mp

    mp_other = MetaPartition(label="label_3", data=pd.DataFrame({"test": [4, 5, 6]}))
    mp_other = mp_1.add_metapartition(mp_other)
    assert mp != mp_other
    assert mp_other != mp


def test_nested_incompatible_meta():
    mp = MetaPartition(
        label="label_1",
        data=pd.DataFrame({"test": np.array([1, 2, 3], dtype=np.int8)}),
        metadata_version=4,
    )

    mp_2 = MetaPartition(
        label="label_2",
        data=pd.DataFrame({"test": np.array([4, 5, 6], dtype=np.float64)}),
        metadata_version=4,
    )
    with pytest.raises(ValueError):
        mp.add_metapartition(mp_2)


def test_unique_label():
    label_list = ["first_0", "first_1"]

    assert _unique_label(label_list) == "first"

    label_list = ["test_0", "test_1"]

    assert _unique_label(label_list) == "test"

    label_list = ["test_0", "test_1", "test_2"]

    assert _unique_label(label_list) == "test"

    label_list = ["something", "else"]

    assert _unique_label(label_list) == "something_else"


def test_merge_indices():
    indices = [
        MetaPartition(
            label="label1",
            indices={"location": {"Loc1": ["label1"], "Loc2": ["label1"]}},
        ),
        MetaPartition(
            label="label2",
            indices={
                "location": {"Loc3": ["label2"], "Loc2": ["label2"]},
                "product": {"Product1": ["label2"], "Product2": ["label2"]},
            },
        ),
    ]
    result = MetaPartition.merge_indices(indices)
    expected = {
        "location": ExplicitSecondaryIndex(
            "location",
            {"Loc1": ["label1"], "Loc2": ["label1", "label2"], "Loc3": ["label2"]},
        ),
        "product": ExplicitSecondaryIndex(
            "product", {"Product1": ["label2"], "Product2": ["label2"]}
        ),
    }
    assert result == expected


def test_build_indices():
    columns = ["location", "product"]
    df = pd.DataFrame(
        OrderedDict(
            [("location", ["Loc1", "Loc2"]), ("product", ["Product1", "Product2"])]
        )
    )
    mp = MetaPartition(label="partition_label", data=df)
    result_mp = mp.build_indices(columns)
    result = result_mp.indices
    loc_index = ExplicitSecondaryIndex(
        "location", {"Loc1": ["partition_label"], "Loc2": ["partition_label"]}
    )
    prod_index = ExplicitSecondaryIndex(
        "product", {"Product1": ["partition_label"], "Product2": ["partition_label"]}
    )

    assert result["location"] == loc_index
    assert result["product"] == prod_index


def test_add_metapartition():
    mp = MetaPartition(
        label="label_1",
        data=pd.DataFrame({"test": [1, 2, 3]}),
        indices={"test": [1, 2, 3]},
    )

    mp_2 = MetaPartition(
        label="label_2",
        data=pd.DataFrame({"test": [4, 5, 6]}),
        indices={"test": [4, 5, 6]},
    )

    new_mp = mp.add_metapartition(mp_2)

    # Cannot access single object attributes
    with pytest.raises(AttributeError):
        new_mp.indices
    with pytest.raises(AttributeError):
        new_mp.label
    with pytest.raises(AttributeError):
        new_mp.data
    with pytest.raises(AttributeError):
        new_mp.file
    with pytest.raises(AttributeError):
        new_mp.indices
    with pytest.raises(AttributeError):
        new_mp.indices

    partition_list = new_mp.metapartitions

    assert len(partition_list) == 2

    first_mp = partition_list[0]
    assert first_mp["label"] == "label_1"
    assert first_mp["indices"] == {"test": [1, 2, 3]}

    first_mp = partition_list[1]
    assert first_mp["label"] == "label_2"
    assert first_mp["indices"] == {"test": [4, 5, 6]}

    # This tests whether it is possible to add to an already nested MetaPartition
    mp_3 = MetaPartition(
        label="label_3",
        data=pd.DataFrame({"test": [7, 8, 9]}),
        indices={"test": [7, 8, 9]},
    )
    new_mp = new_mp.add_metapartition(mp_3)

    partition_list = new_mp.metapartitions

    assert len(partition_list) == 3

    first_mp = partition_list[0]
    assert first_mp["label"] == "label_1"
    assert first_mp["indices"] == {"test": [1, 2, 3]}

    first_mp = partition_list[1]
    assert first_mp["label"] == "label_2"
    assert first_mp["indices"] == {"test": [4, 5, 6]}

    first_mp = partition_list[2]
    assert first_mp["label"] == "label_3"
    assert first_mp["indices"] == {"test": [7, 8, 9]}


def test_to_dict(metadata_version):
    df = pd.DataFrame({"A": [1]})
    schema = make_meta(df, origin="test")
    mp = MetaPartition(
        label="label_1",
        file="file",
        data=df,
        indices={"test": [1, 2, 3]},
        metadata_version=metadata_version,
        schema=schema,
    )
    mp_dct = mp.to_dict()
    assert mp_dct == {
        "label": "label_1",
        "data": df,
        "file": "file",
        "indices": {"test": [1, 2, 3]},
        "metadata_version": metadata_version,
        "schema": schema,
        "partition_keys": [],
        "logical_conjunction": None,
        "table_name": SINGLE_TABLE,
    }


def test_add_metapartition_duplicate_labels():
    mp = MetaPartition(label="label")

    mp_2 = MetaPartition(label="label")
    with pytest.raises(RuntimeError):
        mp.add_metapartition(mp_2)


def test_copy():
    mp = MetaPartition(
        label="label_1", file="file", data=pd.DataFrame(), indices={"test": [1, 2, 3]}
    )
    new_mp = mp.copy()

    # Check if the copy is identical
    assert new_mp == mp
    # ... but not the same object
    assert id(new_mp) != id(mp)

    new_mp = mp.copy(file="new_file")
    assert id(new_mp) != id(mp)
    assert new_mp.file == "new_file"

    new_mp = mp.copy(indices={"new": [1, 2, 3]})
    assert id(new_mp) != id(mp)
    assert new_mp.indices == {"new": [1, 2, 3]}


def test_nested_copy():
    mp = MetaPartition(
        label="label_1",
        file="file",
        data=pd.DataFrame({"test": [1, 2, 3]}),
        indices={"test": {1: "label_1", 2: "label_2", 3: "label_3"}},
    )

    mp_2 = MetaPartition(
        label="label_2",
        data=pd.DataFrame({"test": [4, 5, 6]}),
        indices={"test": [4, 5, 6]},
    )
    mp = mp.add_metapartition(mp_2)
    assert len(mp.metapartitions) == 2
    new_mp = mp.copy()

    # Check if the copy is identical
    assert len(new_mp.metapartitions) == len(mp.metapartitions)
    assert new_mp == mp
    # ... but not the same object
    assert id(new_mp) != id(mp)


def test_partition_on_one_level():
    original_df = pd.DataFrame({"test": [1, 2, 3], "some_values": [1, 2, 3]})
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )

    new_mp = mp.partition_on(["test"])

    assert len(new_mp.metapartitions) == 3

    labels = set()
    for mp in new_mp:
        labels.add(mp.label)
        assert len(mp.data) == 1
        assert mp.data is not None
        df = mp.data
        assert df._is_view

        # try to be agnostic about the order
        assert len(df) == 1
        assert "test" not in df
    expected_labels = set(["test=1/label_1", "test=2/label_1", "test=3/label_1"])
    assert labels == expected_labels


def test_partition_on_one_level_ts():
    original_df = pd.DataFrame(
        {
            "test": [
                pd.Timestamp("2001-01-01"),
                pd.Timestamp("2001-01-02"),
                pd.Timestamp("2001-01-03"),
            ],
            "some_values": [1, 2, 3],
        }
    )
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )

    new_mp = mp.partition_on(["test"])

    assert len(new_mp.metapartitions) == 3

    labels = set()
    for mp in new_mp:
        labels.add(mp.label)
        assert len(mp.data) == 1
        assert mp.data is not None
        df = mp.data
        assert df._is_view

        # try to be agnostic about the order
        assert len(df) == 1
        assert "test" not in df
    expected_labels = set(
        [
            "test=2001-01-01%2000%3A00%3A00/label_1",
            "test=2001-01-02%2000%3A00%3A00/label_1",
            "test=2001-01-03%2000%3A00%3A00/label_1",
        ]
    )
    assert labels == expected_labels


def test_partition_on_roundtrip(store):
    original_df = pd.DataFrame(
        OrderedDict([("test", [1, 2, 3]), ("some_values", [1, 2, 3])])
    )
    mp = MetaPartition(label="label_1", data=original_df, metadata_version=4)

    new_mp = mp.partition_on(["test"])
    new_mp = new_mp.store_dataframes(store=store, dataset_uuid="some_uuid")
    store_schema_metadata(new_mp.schema, "some_uuid", store)
    # Test immediately after dropping and later once with new metapartition to check table meta reloading
    new_mp = new_mp.load_dataframes(store=store)
    assert len(new_mp.metapartitions) == 3
    dfs = []
    for internal_mp in new_mp:
        dfs.append(internal_mp.data)
    actual_df = pd.concat(dfs).sort_values(by="test").reset_index(drop=True)
    pdt.assert_frame_equal(original_df, actual_df)

    for i in range(1, 4):
        # Check with fresh metapartitions
        new_mp = MetaPartition(
            label=f"test={i}/label_1",
            file=f"some_uuid/table/test={i}/label_1.parquet",
            metadata_version=4,
        )
        new_mp = new_mp.load_dataframes(store=store)

        actual_df = new_mp.data

        expected_df = pd.DataFrame(OrderedDict([("test", [i]), ("some_values", [i])]))
        pdt.assert_frame_equal(expected_df, actual_df)


@pytest.mark.parametrize("empty", [True, False])
def test_partition_on_raises_no_cols_left(empty):
    original_df = pd.DataFrame({"test": [1, 2, 3]})
    if empty:
        original_df = original_df.loc[[]]
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )
    with pytest.raises(ValueError) as e:
        mp.partition_on(["test"])
    assert str(e.value) == "No data left to save outside partition columns"


@pytest.mark.parametrize("empty", [True, False])
def test_partition_on_raises_pocols_missing(empty):
    original_df = pd.DataFrame({"test": [1, 2, 3]})
    if empty:
        original_df = original_df.loc[[]]
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )
    with pytest.raises(ValueError) as e:
        mp.partition_on(["test", "foo", "bar"])
    assert str(e.value) == "Partition column(s) missing: bar, foo"


def test_partition_urlencode():
    original_df = pd.DataFrame({"ÖŒå": [1, 2, 3], "some_values": [1, 2, 3]})
    mp = MetaPartition(label="label_1", data=original_df, metadata_version=4)

    new_mp = mp.partition_on(["ÖŒå"])

    assert len(new_mp.metapartitions) == 3

    labels = set()
    for mp in new_mp:
        labels.add(mp.label)
        assert len(mp.data) == 1
        assert mp.data is not None
        df = mp.data
        assert df._is_view

        # try to be agnostic about the order
        assert len(df) == 1
        assert "ÖŒå" not in df
    expected_labels = set(
        [
            "%C3%96%C5%92%C3%A5=1/label_1",
            "%C3%96%C5%92%C3%A5=2/label_1",
            "%C3%96%C5%92%C3%A5=3/label_1",
        ]
    )
    assert labels == expected_labels


def test_partition_two_level():
    original_df = pd.DataFrame(
        {
            "level1": [1, 2, 3, 1, 2, 3],
            "level2": [1, 1, 1, 2, 2, 2],
            "no_index_col": np.arange(0, 6),
        }
    )
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )

    new_mp = mp.partition_on(["level1", "level2"])
    assert len(new_mp.metapartitions) == 6

    labels = []
    for mp in new_mp:
        labels.append(mp.label)
        assert len(mp.data) == 1
        assert mp.data is not None
        df = mp.data
        assert df._is_view

        # try to be agnostic about the order
        assert len(df) == 1
        assert "level1" not in df
        assert "level2" not in df
        assert "no_index_col" in df
    expected_labels = [
        "level1=1/level2=1/label_1",
        "level1=1/level2=2/label_1",
        "level1=2/level2=1/label_1",
        "level1=2/level2=2/label_1",
        "level1=3/level2=1/label_1",
        "level1=3/level2=2/label_1",
    ]
    assert sorted(labels) == sorted(expected_labels)


def test_partition_on_nested():
    original_df = pd.DataFrame(
        {
            "level1": [1, 2, 3, 1, 2, 3],
            "level2": [1, 1, 1, 2, 2, 2],
            "no_index_col": np.arange(0, 6),
        }
    )
    mp = MetaPartition(
        label="label_1", file="file", data=original_df, metadata_version=4
    )
    mp2 = MetaPartition(
        label="label_2", file="file", data=original_df, metadata_version=4
    )
    mp = mp.add_metapartition(mp2)
    new_mp = mp.partition_on(["level1", "level2"])
    assert len(new_mp.metapartitions) == 12

    labels = []
    for mp in new_mp:
        labels.append(mp.label)
        assert len(mp.data) == 1
        assert mp.data is not None
        df = mp.data
        assert df._is_view

        # try to be agnostic about the order
        assert len(df) == 1
        assert "level1" not in df
        assert "level2" not in df
        assert "no_index_col" in df
    expected_labels = [
        "level1=1/level2=1/label_1",
        "level1=1/level2=2/label_1",
        "level1=2/level2=1/label_1",
        "level1=2/level2=2/label_1",
        "level1=3/level2=1/label_1",
        "level1=3/level2=2/label_1",
        "level1=1/level2=1/label_2",
        "level1=1/level2=2/label_2",
        "level1=2/level2=1/label_2",
        "level1=2/level2=2/label_2",
        "level1=3/level2=1/label_2",
        "level1=3/level2=2/label_2",
    ]
    assert sorted(labels) == sorted(expected_labels)


def test_partition_on_stable_order():
    """
    Assert that the partition_on algo is stable wrt to row ordering
    """
    unique_values = 3
    total_values = 20
    random_index = np.repeat(
        np.arange(unique_values), int(np.ceil(total_values / unique_values))
    )[:total_values]
    np.random.shuffle(random_index)
    df = pd.DataFrame(
        {"partition_key": random_index, "sorted_col": range(total_values)}
    )
    mp = MetaPartition(label="label_1", data=df, metadata_version=4)
    new_mp = mp.partition_on("partition_key")
    for sub_mp in new_mp:
        sub_df = sub_mp.data
        assert sub_df.sorted_col.is_monotonic


def test_table_meta(store):
    mp = MetaPartition(
        label="label_1",
        data=pd.DataFrame(
            {
                "i32": np.array([1, 2, 3, 1, 2, 3], dtype="int32"),
                "float": np.array([1, 1, 1, 2, 2, 2], dtype="float64"),
            }
        ),
        metadata_version=4,
    )

    assert mp.schema is not None
    expected_meta = make_meta(
        pd.DataFrame(
            {"i32": np.array([], dtype="int32"), "float": np.array([], dtype="float64")}
        ),
        origin="1",
    )
    actual_meta = mp.schema
    assert actual_meta == expected_meta

    mp = mp.store_dataframes(store, "dataset_uuid")

    actual_meta = mp.schema
    assert actual_meta == expected_meta


def test_partition_on_explicit_index():
    original_df = pd.DataFrame(
        {
            "level1": [1, 2, 1, 2, 1, 2],
            "level2": [1, 1, 1, 2, 2, 2],
            "explicit_index_col": np.arange(0, 6),
        }
    )
    mp = MetaPartition(
        label="label_1",
        file="file",
        data=original_df,
        indices={
            "explicit_index_col": {value: ["label_1"] for value in np.arange(0, 6)}
        },
        metadata_version=4,
    )
    new_mp = mp.partition_on(["level1", "level2"])
    assert len(new_mp) == 4

    expected_indices = {
        "explicit_index_col": ExplicitSecondaryIndex(
            "explicit_index_col",
            {0: ["level1=1/level2=1/label_1"], 2: ["level1=1/level2=1/label_1"]},
        )
    }
    assert expected_indices == new_mp["level1=1/level2=1/label_1"].indices

    expected_indices = {
        "explicit_index_col": ExplicitSecondaryIndex(
            "explicit_index_col", {4: ["level1=1/level2=2/label_1"]}
        )
    }
    assert expected_indices == new_mp["level1=1/level2=2/label_1"].indices

    expected_indices = {
        "explicit_index_col": ExplicitSecondaryIndex(
            "explicit_index_col", {1: ["level1=2/level2=1/label_1"]}
        )
    }
    assert expected_indices == new_mp["level1=2/level2=1/label_1"].indices

    expected_indices = {
        "explicit_index_col": ExplicitSecondaryIndex(
            "explicit_index_col",
            {3: ["level1=2/level2=2/label_1"], 5: ["level1=2/level2=2/label_1"]},
        )
    }
    assert expected_indices == new_mp["level1=2/level2=2/label_1"].indices


def test_reconstruct_index_duplicates(store):
    ser = ParquetSerializer()
    df = pd.DataFrame({"index_col": [1, 1], "column": list("ab")})

    label = "dontcare"
    key_prefix = "uuid/table/index_col=2/{}".format(label)
    key = ser.store(store, key_prefix, df)

    schema = make_meta(df, origin="1", partition_keys="index_col")
    store_schema_metadata(schema, "uuid", store)

    mp = MetaPartition(
        label="dontcare",
        file=key,
        metadata_version=4,
        schema=schema,
        partition_keys=["index_col"],
    )
    mp = mp.load_dataframes(store)
    df_actual = mp.data
    df_expected = pd.DataFrame(
        OrderedDict([("index_col", [2, 2]), ("column", list("ab"))])
    )
    pdt.assert_frame_equal(df_actual, df_expected)


def test_reconstruct_index_categories(store):
    ser = ParquetSerializer()
    df = pd.DataFrame(
        {"index_col": [1, 1], "second_index_col": [2, 2], "column": list("ab")}
    )

    label = "dontcare"
    key_prefix = "uuid/table/index_col=2/second_index_col=2/{}".format(label)
    key = ser.store(store, key_prefix, df)

    schema = make_meta(df, origin="1", partition_keys="index_col")
    store_schema_metadata(schema, "uuid", store)

    mp = MetaPartition(
        label="index_col=2/dontcare",
        file=key,
        metadata_version=4,
        schema=schema,
        partition_keys=["index_col", "second_index_col"],
    )
    categories = ["second_index_col", "index_col"]
    mp = mp.load_dataframes(store, categoricals=categories)
    df_actual = mp.data
    df_expected = pd.DataFrame(
        OrderedDict(
            [
                ("index_col", [2, 2]),
                ("second_index_col", [2, 2]),
                ("column", list("ab")),
            ]
        )
    )
    df_expected = df_expected.astype({col: "category" for col in categories})
    pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("categoricals", [True, False])
def test_reconstruct_index_empty_df(store, categoricals):
    ser = ParquetSerializer()
    df = pd.DataFrame({"index_col": [1, 1], "column": list("ab")})
    df = df[0:0]

    label = "dontcare"
    key_prefix = "uuid/table/index_col=2/{}".format(label)
    key = ser.store(store, key_prefix, df)

    schema = make_meta(df, origin="1", partition_keys="index_col")
    store_schema_metadata(schema, "uuid", store)

    mp = MetaPartition(
        label="index_col=2/dontcare",
        file=key,
        metadata_version=4,
        schema=schema,
        partition_keys=["index_col"],
    )
    categoricals = None
    if categoricals:
        categoricals = ["index_col"]
    mp = mp.load_dataframes(store, categoricals=categoricals)
    df_actual = mp.data
    df_expected = pd.DataFrame(
        OrderedDict([("index_col", [2, 2]), ("column", list("ab"))])
    )
    df_expected = df_expected[0:0]
    if categoricals:
        df_expected = df_expected.astype({"index_col": "category"})
    pdt.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("dates_as_object", [True, False])
def test_reconstruct_date_index(store, metadata_version, dates_as_object):
    ser = ParquetSerializer()
    # If the parquet file does include the primary index col, still use the reconstructed index and ignore the content of the file
    df = pd.DataFrame(
        {"index_col": [date(2018, 6, 1), date(2018, 6, 1)], "column": list("ab")}
    )

    label = "dontcare"
    key_prefix = "uuid/table/index_col=2018-06-02/{}".format(label)
    key = ser.store(store, key_prefix, df)

    schema = make_meta(df, origin="1", partition_keys="index_col")
    store_schema_metadata(schema, "uuid", store)

    mp = MetaPartition(
        label="dontcare",
        file=key,
        metadata_version=metadata_version,
        schema=schema,
        partition_keys=["index_col"],
    )

    mp = mp.load_dataframes(store, dates_as_object=dates_as_object)
    df_actual = mp.data
    if dates_as_object:
        dt_constructor = date
    else:
        dt_constructor = datetime
    df_expected = pd.DataFrame(
        OrderedDict(
            [
                ("index_col", [dt_constructor(2018, 6, 2), dt_constructor(2018, 6, 2)]),
                ("column", list("ab")),
            ]
        )
    )
    pdt.assert_frame_equal(df_actual, df_expected)


def test_iter_empty_metapartition():
    for mp in MetaPartition(None):
        raise AssertionError(
            "Iterating over an empty MetaPartition should stop immediately"
        )


def test_concat_metapartition(df_all_types):
    mp1 = MetaPartition(label="first", data=df_all_types, metadata_version=4)
    mp2 = MetaPartition(label="second", data=df_all_types, metadata_version=4)

    new_mp = MetaPartition.concat_metapartitions([mp1, mp2])

    # what the label actually is, doesn't matter so much
    assert new_mp.label is not None
    df_expected = pd.concat([df_all_types, df_all_types])
    df_actual = new_mp.data
    pdt.assert_frame_equal(df_actual, df_expected)


def test_concat_metapartition_wrong_types(df_all_types):
    mp1 = MetaPartition(label="first", data=df_all_types, metadata_version=4)
    df_corrupt = df_all_types.copy()
    df_corrupt["int8"] = "NoInteger"
    mp2 = MetaPartition(label="second", data=df_corrupt, metadata_version=4)

    with pytest.raises(ValueError, match="Schema violation"):
        MetaPartition.concat_metapartitions([mp1, mp2])


def test_concat_metapartition_partitioned(df_all_types):
    mp1 = MetaPartition(
        label="int8=1/1234",
        data=df_all_types,
        metadata_version=4,
        partition_keys=["int8"],
    )
    mp2 = MetaPartition(
        label="int8=1/4321",
        data=df_all_types,
        metadata_version=4,
        partition_keys=["int8"],
    )

    new_mp = MetaPartition.concat_metapartitions([mp1, mp2])

    df_expected = pd.concat([df_all_types, df_all_types])
    df_actual = new_mp.data
    pdt.assert_frame_equal(df_actual, df_expected)

    assert new_mp.partition_keys == ["int8"]


def test_concat_metapartition_different_partitioning(df_all_types):
    mp1 = MetaPartition(
        label="int8=1/1234",
        data=df_all_types,
        metadata_version=4,
        partition_keys=["int8"],
    )
    mp2 = MetaPartition(
        label="float8=1.0/4321",
        data=df_all_types,
        metadata_version=4,
        partition_keys=["float8"],
    )

    with pytest.raises(ValueError, match="Schema violation"):
        MetaPartition.concat_metapartitions([mp1, mp2])


def test_concat_metapartition_categoricals(df_all_types):
    mp1 = MetaPartition(
        label="first",
        data=pd.DataFrame({"a": [0, 0], "b": ["a", "a"]}, dtype="category"),
        metadata_version=4,
        partition_keys=["a"],
    )
    mp2 = MetaPartition(
        label="second",
        data=pd.DataFrame({"a": [1, 1], "b": ["a", "b"]}, dtype="category"),
        metadata_version=4,
        partition_keys=["a"],
    )

    new_mp = MetaPartition.concat_metapartitions([mp1, mp2])

    assert new_mp.table_name == "table"
    assert pd.api.types.is_categorical_dtype(new_mp.data["b"].dtype)


# We can't partition on null columns (gh-262)
@pytest.mark.parametrize(
    "col", sorted(set(get_dataframe_not_nested().columns) - {"null"})
)
def test_partition_on_scalar_intermediate(df_not_nested, col):
    """
    Test against a bug where grouping leaves a scalar value
    """
    assert len(df_not_nested) == 1
    mp = MetaPartition(label="somelabel", data=df_not_nested, metadata_version=4)
    new_mp = mp.partition_on(col)
    assert len(new_mp) == 1


def test_partition_on_with_primary_index_invalid(df_not_nested):
    mp = MetaPartition(
        label="pkey=1/pkey2=2/base_label",
        data=df_not_nested,
        partition_keys=["pkey", "pkey2"],
        metadata_version=4,
    )
    with pytest.raises(ValueError, match="Incompatible"):
        mp.partition_on("int64")

    with pytest.raises(ValueError, match="Incompatible"):
        mp.partition_on(["int64", "pkey"])

    with pytest.raises(ValueError, match="Incompatible"):
        mp.partition_on(["pkey", "int64"])

    with pytest.raises(ValueError, match="Incompatible"):
        mp.partition_on(["pkey2", "pkey1", "int64"])

    mp.partition_on(["pkey", "pkey2"])
    mp.partition_on(["pkey", "pkey2", "int64"])


def test_partition_on_with_primary_index(df_not_nested):
    mp = MetaPartition(
        label="pkey=1/base_label",
        data=df_not_nested,
        partition_keys=["pkey"],
        metadata_version=4,
    )
    new = mp.partition_on(["pkey", "int64"])

    split_label = new.label.split("/")

    assert len(split_label) == 3
    assert split_label[0] == "pkey=1"
    assert split_label[1] == "int64=1"
    assert split_label[2] == "base_label"

    assert mp == mp.partition_on(["pkey"])


@pytest.mark.parametrize(
    "labels, flat_labels",
    [
        ([], []),
        (["a"], ["a"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", None], ["a"]),
        (["a", ["b", "c"]], ["a", "b", "c"]),
        (["a", None, ["b", "c", None]], ["a", "b", "c"]),
    ],
)
def test_partition_label_helper(labels, flat_labels):
    mps = []
    for lbl in labels:
        if isinstance(lbl, list):
            mp = MetaPartition(lbl[0])
            for nested_lbl in lbl[1:]:
                mp = mp.add_metapartition(MetaPartition(label=nested_lbl))
            mps.append(mp)
        else:
            mps.append(MetaPartition(label=lbl))

    assert set(partition_labels_from_mps(mps)) == set(flat_labels)


def test_column_string_cast(df_all_types, store, metadata_version):
    original_columns = df_all_types.columns.copy()
    df_all_types.columns = df_all_types.columns.str.encode("utf-8")
    ser = ParquetSerializer()
    key = ser.store(store, "uuid/table/something", df_all_types)
    mp = MetaPartition(
        label="something",
        file=key,
        schema=make_meta(df_all_types, origin="table"),
        metadata_version=metadata_version,
    )
    mp = mp.load_dataframes(store)
    df = mp.data
    assert all(original_columns == df.columns)


def test_partition_on_valid_schemas():
    """
    Ensure that partitioning is possible even if the output schemas of the
    sub partitions may be different
    """
    df = pd.DataFrame({"partition_col": [0, 1], "values": [None, "str"]})
    mp = MetaPartition(label="base_label", data=df, metadata_version=4)
    mp = mp.partition_on(["partition_col"])
    assert len(mp) == 2
    expected_meta = make_meta(df, origin="1", partition_keys="partition_col")
    assert mp.schema == expected_meta


def test_input_to_metaframes_empty():
    mp = parse_input_to_metapartition(obj=[None])
    assert mp == MetaPartition(label=None)
    mp = parse_input_to_metapartition(obj=[])
    assert mp == MetaPartition(label=None)


def test_input_to_metaframes_simple():
    df_input = pd.DataFrame({"A": [1]})
    mp = parse_input_to_metapartition(obj=df_input)

    assert isinstance(mp, MetaPartition)
    assert len(mp.data) == 1
    assert mp.file is None

    df = mp.data
    pdt.assert_frame_equal(df, df_input)

    assert isinstance(mp.label, str)


def test_parse_nested_input_schema_compatible_but_different():
    # Ensure that input can be parsed even though the schemas are not identical but compatible
    df_input = [[pd.DataFrame({"A": [None]}), pd.DataFrame({"A": ["str"]})]]
    mp = parse_input_to_metapartition(df_input, metadata_version=4)
    expected_schema = make_meta(pd.DataFrame({"A": ["str"]}), origin="expected")
    assert mp.schema == expected_schema


def test_get_parquet_metadata(store):
    df = pd.DataFrame({"P": np.arange(0, 10), "L": np.arange(0, 10)})
    mp = MetaPartition(label="test_label", data=df)
    meta_partition = mp.store_dataframes(store=store, dataset_uuid="dataset_uuid")

    actual = meta_partition.get_parquet_metadata(store=store)
    actual.drop(labels="serialized_size", axis=1, inplace=True)
    actual.drop(labels="row_group_compressed_size", axis=1, inplace=True)
    actual.drop(labels="row_group_uncompressed_size", axis=1, inplace=True)

    expected = pd.DataFrame(
        {
            "partition_label": ["test_label"],
            "row_group_id": 0,
            "number_rows_total": 10,
            "number_row_groups": 1,
            "number_rows_per_row_group": 10,
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_get_parquet_metadata_empty_df(store):
    df = pd.DataFrame()
    mp = MetaPartition(label="test_label", data=df)
    meta_partition = mp.store_dataframes(store=store, dataset_uuid="dataset_uuid")

    actual = meta_partition.get_parquet_metadata(store=store)
    actual.drop(
        columns=[
            "serialized_size",
            "row_group_compressed_size",
            "row_group_uncompressed_size",
        ],
        axis=1,
        inplace=True,
    )

    expected = pd.DataFrame(
        {
            "partition_label": ["test_label"],
            "row_group_id": 0,
            "number_rows_total": 0,
            "number_row_groups": 1,
            "number_rows_per_row_group": 0,
        }
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_get_parquet_metadata_row_group_size(store):
    df = pd.DataFrame({"P": np.arange(0, 10), "L": np.arange(0, 10)})
    mp = MetaPartition(label="test_label", data=df)
    ps = ParquetSerializer(chunk_size=5)

    meta_partition = mp.store_dataframes(
        store=store, dataset_uuid="dataset_uuid", df_serializer=ps
    )
    actual = meta_partition.get_parquet_metadata(store=store)
    actual.drop(
        columns=[
            "serialized_size",
            "row_group_compressed_size",
            "row_group_uncompressed_size",
        ],
        axis=1,
        inplace=True,
    )

    expected = pd.DataFrame(
        {
            "partition_label": ["test_label", "test_label"],
            "row_group_id": [0, 1],
            "number_rows_total": [10, 10],
            "number_row_groups": [2, 2],
            "number_rows_per_row_group": [5, 5],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test__reconstruct_index_columns():
    df = pd.DataFrame({"x": [0], "a": [-1], "b": [-2], "c": [-3]})
    mp = MetaPartition(label="test_label", data=df)
    df_with_index_columns = mp._reconstruct_index_columns(
        df=df[["x"]],
        key_indices=[("a", 1), ("b", 2), ("c", 3)],
        columns=["x", "c"],
        categories=None,
        date_as_object=False,
    )
    # Index columns first
    pdt.assert_frame_equal(df_with_index_columns, pd.DataFrame({"c": [3], "x": [0]}))


def test_partition_on_keeps_table_name():
    mp = MetaPartition(
        label="label_1",
        data=pd.DataFrame({"P": [1, 2, 1, 2], "L": [1, 1, 2, 2]}),
        table_name="non-default-name",
    )
    repartitioned_mp = mp.partition_on(["P"])
    assert repartitioned_mp.table_name == "non-default-name"
