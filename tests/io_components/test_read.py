import types
from collections import OrderedDict

import pandas as pd
import pytest

from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io_components.metapartition import MetaPartition
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

    assert set(mp.table_meta.keys()) == {"core", "helper"}


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


@pytest.mark.parametrize("predicates", [[], [[]]])
def test_dispatch_metapartition_undefined_behaviour(dataset, store_session, predicates):
    with pytest.raises(ValueError) as exc:
        list(
            dispatch_metapartitions(dataset.uuid, store_session, predicates=predicates)
        )
    assert "The behaviour on an empty" in str(exc.value)


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
        [[("P", "==", 2)], [("TARGET", "==", 500)]],
        [[("P", "in", [2])], [("TARGET", "in", [500])]],
        [[("L", "==", 2)], [("TARGET", "==", 500)]],
    ],
)
def test_dispatch_metapartitions_query_no_effect(
    dataset_partition_keys, store_session, predicates
):
    # These predicates should still lead to loading the whole set of partitionss
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

    mps = list(
        dispatch_metapartitions(
            dataset.uuid, store, concat_partitions_on_primary_index=True
        )
    )
    assert len(mps) == 1
