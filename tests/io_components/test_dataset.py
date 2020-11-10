"""

This module tests high level dataset API functions which require entire datasets, indices, etc

"""
from collections import OrderedDict

import pandas as pd
import pandas.testing as pdt

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex


def test_dataset_get_indices_as_dataframe_partition_keys_only(
    dataset_with_index, store_session
):
    expected = pd.DataFrame(
        OrderedDict([("P", [1, 2])]),
        index=pd.Index(["P=1/cluster_1", "P=2/cluster_2"], name="partition"),
    )
    ds = dataset_with_index.load_partition_indices()
    result = ds.get_indices_as_dataframe(columns=dataset_with_index.partition_keys)
    pdt.assert_frame_equal(result, expected)


def test_dataset_get_indices_as_dataframe(dataset_with_index, store_session):
    expected = pd.DataFrame(
        OrderedDict([("L", [1, 2]), ("P", [1, 2])]),
        index=pd.Index(["P=1/cluster_1", "P=2/cluster_2"], name="partition"),
    )
    ds = dataset_with_index.load_partition_indices()
    ds = ds.load_index("L", store_session)

    result = ds.get_indices_as_dataframe()
    pdt.assert_frame_equal(result, expected)


def test_dataset_get_indices_as_dataframe_duplicates():
    ds = DatasetMetadata(
        "some_uuid",
        indices={
            "l_external_code": ExplicitSecondaryIndex(
                "l_external_code", {"1": ["part1", "part2"], "2": ["part1", "part2"]}
            ),
            "p_external_code": ExplicitSecondaryIndex(
                "p_external_code", {"1": ["part1"], "2": ["part2"]}
            ),
        },
    )
    expected = pd.DataFrame(
        OrderedDict(
            [
                ("p_external_code", ["1", "1", "2", "2"]),
                ("l_external_code", ["1", "2", "1", "2"]),
            ]
        ),
        index=pd.Index(["part1", "part1", "part2", "part2"], name="partition"),
    )
    result = ds.get_indices_as_dataframe()
    pdt.assert_frame_equal(result, expected)


def test_dataset_get_indices_as_dataframe_predicates():
    ds = DatasetMetadata(
        "some_uuid",
        indices={
            "l_external_code": ExplicitSecondaryIndex(
                "l_external_code", {"1": ["part1", "part2"], "2": ["part1", "part2"]}
            ),
            "p_external_code": ExplicitSecondaryIndex(
                "p_external_code", {"1": ["part1"], "2": ["part2"]}
            ),
        },
    )
    expected = pd.DataFrame(
        OrderedDict([("p_external_code", ["1"])]),
        index=pd.Index(["part1"], name="partition"),
    )
    result = ds.get_indices_as_dataframe(
        columns=["p_external_code"], predicates=[[("p_external_code", "==", "1")]]
    )
    pdt.assert_frame_equal(result, expected)

    result = ds.get_indices_as_dataframe(
        columns=["l_external_code"], predicates=[[("l_external_code", "==", "1")]]
    )
    expected = pd.DataFrame(
        OrderedDict([("l_external_code", "1")]),
        index=pd.Index(["part1", "part2"], name="partition"),
    )
    pdt.assert_frame_equal(result, expected)

    result = ds.get_indices_as_dataframe(
        columns=["l_external_code"],
        predicates=[[("l_external_code", "==", "1"), ("p_external_code", "==", "1")]],
    )
    expected = pd.DataFrame(
        OrderedDict([("l_external_code", "1")]),
        index=pd.Index(["part1"], name="partition"),
    )
    pdt.assert_frame_equal(result, expected)

    result = ds.get_indices_as_dataframe(
        columns=["l_external_code"],
        predicates=[[("l_external_code", "==", "1"), ("p_external_code", "==", "3")]],
    )
    expected = pd.DataFrame(
        columns=["l_external_code"], index=pd.Index([], name="partition")
    )
    pdt.assert_frame_equal(result, expected)


def test_dataset_get_indices_as_dataframe_no_index(dataset):
    assert not dataset.primary_indices_loaded
    df = dataset.get_indices_as_dataframe()
    pdt.assert_frame_equal(df, pd.DataFrame(index=["cluster_1", "cluster_2"]))


def test_dataset_get_indices_as_dataframe_with_index(dataset_with_index, store_session):
    assert not dataset_with_index.primary_indices_loaded
    df = dataset_with_index.get_indices_as_dataframe()
    pdt.assert_frame_equal(
        df,
        pd.DataFrame(columns=["L", "P"], index=pd.Index([], name="partition")).astype(
            {"L": "object", "P": "int64"}
        ),
    )
