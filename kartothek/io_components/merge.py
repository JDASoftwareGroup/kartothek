import logging
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, cast

import pandas as pd

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.factory import DatasetFactory
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.utils import _instantiate_store, _make_callable

if TYPE_CHECKING:
    from simplekv import KeyValueStore


LOGGER = logging.getLogger(__name__)


def align_datasets(left_dataset_uuid, right_dataset_uuid, store, match_how="exact"):
    """
    Determine dataset partition alignment

    Parameters
    ----------
    left_dataset_uuid : basestring
    right_dataset_uuid : basestring
    store : KeyValuestore or callable
    match_how : basestring or callable, {exact, prefix, all, callable}

    Yields
    ------
    list
    """
    store = _instantiate_store(store)
    left_dataset = DatasetMetadata.load_from_store(uuid=left_dataset_uuid, store=store)
    right_dataset = DatasetMetadata.load_from_store(
        uuid=right_dataset_uuid, store=store
    )

    metadata_version = left_dataset.metadata_version

    # Loop over the dataset with fewer partitions, treating its keys as
    # partition label prefixes
    if (
        callable(match_how)
        or match_how == "left"
        or (
            match_how == "prefix"
            and len(list(left_dataset.partitions.keys())[0])
            < len(list(right_dataset.partitions.keys())[0])
        )
    ):
        first_dataset = left_dataset
        second_dataset = right_dataset
    else:
        first_dataset = right_dataset
        second_dataset = left_dataset
    # The del statements are here to reduce confusion below
    del left_dataset
    del right_dataset

    # For every partition in the 'small' dataset, at least one partition match
    # needs to be found in the larger dataset.
    available_partitions = list(second_dataset.partitions.items())
    partition_stack = available_partitions[:]

    # TODO: write a test which protects against the following scenario!!
    # Sort the partition labels by length of the labels, starting with the
    # labels which are the longest. This way we prevent label matching for
    # similar partitions, e.g. cluster_100 and cluster_1. This, of course,
    # works only as long as the internal loop removes elements which were
    # matched already (here improperly called stack)
    for l_1 in sorted(first_dataset.partitions, key=len, reverse=True):
        p_1 = first_dataset.partitions[l_1]
        res = [
            MetaPartition.from_partition(
                partition=p_1, metadata_version=metadata_version
            )
        ]
        for parts in available_partitions:
            l_2, p_2 = parts
            if callable(match_how) and not match_how(l_1, l_2):
                continue
            if match_how == "exact" and l_1 != l_2:
                continue
            elif match_how == "prefix" and not l_2.startswith(l_1):
                LOGGER.debug("rejecting (%s, %s)", l_1, l_2)
                continue

            LOGGER.debug(
                "Found alignment between partitions " "(%s, %s) and" "(%s, %s)",
                first_dataset.uuid,
                p_1.label,
                second_dataset.uuid,
                p_2.label,
            )
            res.append(
                MetaPartition.from_partition(
                    partition=p_2, metadata_version=metadata_version
                )
            )

            # In exact or prefix matching schemes, it is expected to only
            # find one partition alignment. in this case reduce the size of
            # the inner loop
            if match_how in ["exact", "prefix"]:
                partition_stack.remove((l_2, p_2))
        # Need to copy, otherwise remove will alter the loop iterator
        available_partitions = partition_stack[:]
        if len(res) == 1:
            raise RuntimeError(
                "No matching partition for {} in dataset {} "
                "found".format(p_1, first_dataset)
            )
        yield res


def align_datasets_many(
    dataset_uuids: List[str],
    store,
    match_how: str = "exact",
    dispatch_by: Optional[List[str]] = None,
    predicates=None,
):
    """
    Determine dataset partition alignment

    Parameters
    ----------
    left_dataset_uuid : basestring
    right_dataset_uuid : basestring
    store : KeyValuestore or callable
    match_how : basestring or callable, {exact, prefix, all, callable}

    Yields
    ------
    list
    """
    if len(dataset_uuids) < 2:
        raise ValueError("Need at least two datasets for merging.")
    dataset_factories = [
        DatasetFactory(
            dataset_uuid=dataset_uuid,
            store_factory=cast(Callable[[], "KeyValueStore"], _make_callable(store)),
            load_schema=True,
            load_all_indices=False,
            load_dataset_metadata=True,
        ).load_partition_indices()
        for dataset_uuid in dataset_uuids
    ]

    store = _instantiate_store(store)
    mps = [
        # TODO: Add predicates
        # We don't pass dispatch_by here as we will do the dispatching later
        list(
            dispatch_metapartitions_from_factory(
                dataset_factory=dataset_factory, predicates=predicates
            )
        )
        for dataset_factory in dataset_factories
    ]

    if match_how == "first":
        if len(set(len(x) for x in mps)) != 1:
            raise RuntimeError("All datasets must have the same number of partitions")
        for mp_0 in mps[0]:
            for other_mps in zip(*mps[1:]):
                yield [mp_0] + list(other_mps)
    elif match_how == "prefix_first":
        # TODO: write a test which protects against the following scenario!!
        # Sort the partition labels by length of the labels, starting with the
        # labels which are the longest. This way we prevent label matching for
        # similar partitions, e.g. cluster_100 and cluster_1. This, of course,
        # works only as long as the internal loop removes elements which were
        # matched already (here improperly called stack)
        for mp_0 in mps[0]:
            res = [mp_0]
            label_0 = mp_0.label
            for dataset_i in range(1, len(mps)):
                for j, mp_i in enumerate(mps[dataset_i]):
                    if mp_i.label.startswith(label_0):
                        res.append(mp_i)
                        del mps[dataset_i][j]
                        break
                else:
                    raise RuntimeError(
                        f"Did not find a matching partition in dataset {dataset_uuids[dataset_i]} for partition {label_0}"
                    )
            yield res
    elif match_how == "exact":
        raise NotImplementedError("exact")
    elif match_how == "dispatch_by":
        index_dfs = []
        for i, factory in enumerate(dataset_factories):
            df = factory.get_indices_as_dataframe(dispatch_by, predicates=predicates)
            index_dfs.append(
                df.reset_index().rename(
                    columns={"partition": f"partition_{i}"}, copy=False
                )
            )
        index_df = reduce(partial(pd.merge, on=dispatch_by), index_dfs)

        mps_by_label: List[Dict[str, MetaPartition]] = []
        for mpx in mps:
            mps_by_label.append({})
            for mp in mpx:
                mps_by_label[-1][mp.label] = mp

        for _, group in index_df.groupby(dispatch_by):
            res_nested: List[List[MetaPartition]] = []
            for i in range(len(dataset_uuids)):
                res_nested.append(
                    [
                        mps_by_label[i][label]
                        for label in group[f"partition_{i}"].unique()
                    ]
                )
            yield res_nested
    else:
        raise NotImplementedError(f"matching with '{match_how}' is not supported")
