from functools import partial
from typing import List, Optional, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.delayed import Delayed

from kartothek.core.factory import DatasetFactory
from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.utils import sort_values_categorical
from kartothek.serialization import DataFrameSerializer

from ._utils import map_delayed

DelayedList = List[Delayed]


def _update_dask_partitions_shuffle(
    ddf: dd.DataFrame,
    table: str,
    secondary_indices: Optional[List[str]],
    metadata_version: int,
    partition_on: List[str],
    ds_factory: DatasetFactory,
    df_serializer: Optional[DataFrameSerializer],
    num_buckets: int,
    sort_partitions_by: Optional[str],
) -> da.Array:
    if ddf.npartitions == 0:
        return ddf

    splits = np.array_split(
        np.arange(ddf.npartitions), min(ddf.npartitions, num_buckets)
    )
    lower_bounds = map(min, splits)
    upper_bounds = map(max, splits)

    final = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        lower_int = cast(int, lower)
        upper_int = cast(int, upper)
        chunk_ddf = ddf.partitions[lower_int : upper_int + 1]
        chunk_ddf = chunk_ddf.groupby(by=partition_on[0])
        chunk_ddf = chunk_ddf.apply(
            partial(
                _store_partition,
                secondary_indices=secondary_indices,
                sort_partitions_by=sort_partitions_by,
                table=table,
                ds_factory=ds_factory,
                partition_on=partition_on,
                df_serializer=df_serializer,
                metadata_version=metadata_version,
            ),
            meta=("MetaPartition", "object"),
        )
        final.append(chunk_ddf)
    return da.concatenate([val.values for val in final])


def _update_dask_partitions_one_to_one(
    delayed_tasks: List[Delayed],
    secondary_indices: Optional[List[str]],
    metadata_version: int,
    partition_on: Optional[List[str]],
    ds_factory: DatasetFactory,
    df_serializer: Optional[DataFrameSerializer],
    sort_partitions_by: Optional[str],
) -> DelayedList:
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
        store=ds_factory.store_factory,
        df_serializer=df_serializer,
        dataset_uuid=ds_factory.dataset_uuid,
    )


def _store_partition(
    df: pd.DataFrame,
    secondary_indices: List[str],
    sort_partitions_by: Optional[str],
    table: Optional[str],
    ds_factory: DatasetFactory,
    partition_on: Optional[List[str]],
    df_serializer: Optional[DataFrameSerializer],
    metadata_version: int,
) -> MetaPartition:
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
        store=ds_factory.store,
        dataset_uuid=ds_factory.dataset_uuid,
        df_serializer=df_serializer,
    )
