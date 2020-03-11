import math
import types
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition
from kartothek.io_components.read import dispatch_metapartitions


def test_dispatch_metapartitions(dataset, store_session):
    part_generator = dispatch_metapartitions(dataset.uuid, store_session)

    assert isinstance(part_generator, types.GeneratorType)
    partitions = OrderedDict([(part.label, part) for part in part_generator])

    assert len(partitions) == 2
    mp = partitions["cluster_1"]
    assert isinstance(mp, MetaPartition)
    assert dict(mp.dataset_metadata) == dict(dataset.metadata)

    mp = partitions["cluster_2"]
    assert isinstance(mp, MetaPartition)
    assert dict(mp.dataset_metadata) == dict(dataset.metadata)

    assert set(mp.table_meta.keys()) == {SINGLE_TABLE, "helper"}


def test_dispatch_metapartitions_label_filter(dataset, store_session):
    def label_filter(part_label):
        return "cluster_1" in part_label

    part_generator = dispatch_metapartitions(
        dataset.uuid, store_session, label_filter=label_filter
    )

    assert isinstance(part_generator, types.GeneratorType)
    partitions = OrderedDict([(part.label, part) for part in part_generator])

    assert len(partitions) == 1
    mp = partitions["cluster_1"]
    assert isinstance(mp, MetaPartition)
    assert dict(mp.dataset_metadata) == dict(dataset.metadata)


def test_dispatch_metapartitions_without_dataset_metadata(dataset, store_session):
    part_generator = dispatch_metapartitions(
        dataset.uuid, store_session, load_dataset_metadata=False
    )

    assert isinstance(part_generator, types.GeneratorType)
    partitions = list(part_generator)

    assert len(partitions) == 2

    mp = partitions[0]
    assert mp.dataset_metadata == {}

    mp = partitions[1]
    assert mp.dataset_metadata == {}


@pytest.mark.parametrize(
    "predicates,error_msg",
    [([], "Empty predicates"), ([[]], "Invalid predicates: Conjunction 0 is empty")],
)
def test_dispatch_metapartition_undefined_behaviour(
    dataset, store_session, predicates, error_msg
):
    with pytest.raises(ValueError, match=error_msg):
        list(
            dispatch_metapartitions(dataset.uuid, store_session, predicates=predicates)
        )


@pytest.mark.parametrize(
    "predicates",
    [
        [[("P", "==", 2)]],
        [[("P", "in", [2])]],
        [[("P", "!=", 1)]],
        [[("P", ">", 1)]],
        # Only apply filter to columns for which we have an index
        [[("P", ">=", 2), ("TARGET", "==", 500)]],
    ],
)
def test_dispatch_metapartitions_query_partition_on(
    dataset_partition_keys, store_session, predicates
):
    generator = dispatch_metapartitions(
        dataset_partition_keys.uuid, store_session, predicates=predicates
    )
    partitions = list(generator)
    assert len(partitions) == 1
    assert partitions[0].label == "P=2/cluster_2"


@pytest.mark.parametrize(
    "predicates",
    [
        # These predicates are OR connected, therefore they need to allow all partitions
        [[("P", "==", 2)], [("TARGET", "==", 500)]],
        [[("P", "in", [2])], [("TARGET", "in", [500])]],
        [[("L", "==", 2)], [("TARGET", "==", 500)]],
    ],
)
def test_dispatch_metapartitions_query_no_effect(
    dataset_partition_keys, store_session, predicates
):
    # These predicates should still lead to loading the whole set of partitions
    generator = dispatch_metapartitions(
        dataset_partition_keys.uuid, store_session, predicates=predicates
    )
    partitions = list(generator)
    assert len(partitions) == 2


def test_dispatch_metapartitions_concat_regression(store):
    dataset = store_dataframes_as_dataset(
        dfs=[pd.DataFrame({"p": [0], "x": [0]}), pd.DataFrame({"p": [0], "x": [1]})],
        dataset_uuid="test",
        store=store,
        partition_on=["p"],
    )

    mps = list(
        dispatch_metapartitions(
            dataset.uuid, store, concat_partitions_on_primary_index=False
        )
    )
    assert len(mps) == 2

    with pytest.deprecated_call():
        mps = list(
            dispatch_metapartitions(
                dataset.uuid, store, concat_partitions_on_primary_index=True
            )
        )
        assert len(mps) == 1

    mps = list(dispatch_metapartitions(dataset.uuid, store, dispatch_by=["p"]))
    assert len(mps) == 1


def test_dispatch_metapartitions_dups_with_predicates(store):
    dataset = store_dataframes_as_dataset(
        dfs=[pd.DataFrame({"p": [0, 1], "x": 0})],
        dataset_uuid="test",
        store=store,
        secondary_indices=["p"],
    )

    wout_preds = list(dispatch_metapartitions(dataset.uuid, store))
    w_preds = list(
        dispatch_metapartitions(dataset.uuid, store, predicates=[[("p", "in", [0, 1])]])
    )

    assert wout_preds == w_preds


def test_dispatch_metapartitions_dups_with_predicates_dispatch_by(store):
    dataset = store_dataframes_as_dataset(
        dfs=[pd.DataFrame({"p": [0, 1], "x": 0})],
        dataset_uuid="test",
        store=store,
        secondary_indices=["p", "x"],
    )

    wout_preds = list(dispatch_metapartitions(dataset.uuid, store, dispatch_by="x"))
    w_preds = list(
        dispatch_metapartitions(
            dataset.uuid, store, predicates=[[("p", "in", [0, 1])]], dispatch_by="x"
        )
    )

    assert wout_preds == w_preds


def test_dispatch_metapartitions_sorted_dispatch_by(store):
    df = pd.DataFrame(
        {"p": np.random.randint(high=100000, low=-100000, size=(100,)), "x": 0}
    )
    # Integers are sorted when using too small values (maybe connected to the
    # singleton implementation of integers in CPython??)
    # Verify this is not happening, otherwise we'll get immediately a sorted
    # index (which is nice in this case but not generally true, of course)
    arr = set(df["p"].unique())
    assert list(arr) != sorted(arr)

    dataset = store_dataframes_as_dataset(
        dfs=[df], dataset_uuid="test", store=store, secondary_indices=["p", "x"]
    )

    wout_preds = list(dispatch_metapartitions(dataset.uuid, store, dispatch_by="p"))
    last = -math.inf
    for mps in wout_preds:
        for mp in mps:
            current = mp.logical_conjunction
            assert len(current) == 1
            current = current[0][2]
            assert current > last
            last = current
