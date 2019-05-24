# -*- coding: utf-8 -*-


from functools import partial

import dask.array as da
import numpy as np

from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.utils import sort_values_categorical

from ._utils import map_delayed


def _update_dask_partitions_shuffle(
    ddf,
    table,
    secondary_indices,
    metadata_version,
    partition_on,
    store_factory,
    df_serializer,
    dataset_uuid,
    num_buckets,
    sort_partitions_by,
):
    splits = np.array_split(
        np.arange(ddf.npartitions), min(ddf.npartitions, num_buckets)
    )
    lower_bounds = map(min, splits)
    upper_bounds = map(max, splits)

    final = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        chunk_ddf = ddf.partitions[lower : upper + 1]
        chunk_ddf = chunk_ddf.groupby(by=partition_on[0])
        chunk_ddf = chunk_ddf.apply(
            partial(
                _store_partition,
                secondary_indices=secondary_indices,
                sort_partitions_by=sort_partitions_by,
                table=table,
                dataset_uuid=dataset_uuid,
                partition_on=partition_on,
                store_factory=store_factory,
                df_serializer=df_serializer,
                metadata_version=metadata_version,
            ),
            meta=("MetaPartition", "object"),
        )
        final.append(chunk_ddf)
    return da.concatenate([val.values for val in final])


def _update_dask_partitions_one_to_one(
    delayed_tasks,
    secondary_indices,
    metadata_version,
    partition_on,
    store_factory,
    df_serializer,
    dataset_uuid,
    sort_partitions_by,
):
    input_to_mps = partial(
        parse_input_to_metapartition,
        metadata_version=metadata_version,
        expected_secondary_indices=secondary_indices,
    )
    mps = map_delayed(delayed_tasks, input_to_mps)

    if sort_partitions_by:
        mps = map_delayed(
            mps,
            MetaPartition.apply,
            partial(sort_values_categorical, column=sort_partitions_by),
        )
    if partition_on:
        mps = map_delayed(mps, MetaPartition.partition_on, partition_on)
    if secondary_indices:
        mps = map_delayed(mps, MetaPartition.build_indices, secondary_indices)

    return map_delayed(
        mps,
        MetaPartition.store_dataframes,
        store=store_factory,
        df_serializer=df_serializer,
        dataset_uuid=dataset_uuid,
    )


def _store_partition(
    df,
    secondary_indices,
    sort_partitions_by,
    table,
    dataset_uuid,
    partition_on,
    store_factory,
    df_serializer,
    metadata_version,
):
    store = store_factory()
    # I don't have access to the group values
    mps = parse_input_to_metapartition(
        {"data": {table: df}}, metadata_version=metadata_version
    )
    # delete reference to enable release after partition_on; before index build
    del df
    if sort_partitions_by:
        mps = mps.apply(partial(sort_values_categorical, column=sort_partitions_by))
    if partition_on:
        mps = mps.partition_on(partition_on)
    if secondary_indices:
        mps = mps.build_indices(secondary_indices)
    return mps.store_dataframes(
        store=store, dataset_uuid=dataset_uuid, df_serializer=df_serializer
    )
