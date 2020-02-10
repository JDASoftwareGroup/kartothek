# -*- coding: utf-8 -*-


from functools import partial
from typing import List

import numpy as np
import pandas as pd

from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.utils import sort_values_categorical

from ._utils import map_delayed

_KTK_HASH_BUCKET = "__KTK_HASH_BUCKET"


def _hash_bucket(df: pd.DataFrame, subset: List[str], num_buckets: int):
    """
    Categorize each row of `df` based on the data in the columns `subset`
    into `num_buckets` values. This is based on `pandas.util.hash_pandas_object`
    """

    if subset is None:
        subset = df.columns
    hash_arr = pd.util.hash_pandas_object(df[subset], index=False)
    buckets = hash_arr % num_buckets

    available_bit_widths = np.array([8, 16, 32, 64])
    mask = available_bit_widths > np.log2(num_buckets)
    bit_width = min(available_bit_widths[mask])
    df[_KTK_HASH_BUCKET] = buckets.astype(f"uint{bit_width}")
    return df


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
    bucket_by,
):
    if ddf.npartitions == 0:
        return ddf

    if num_buckets is not None:
        meta = ddf._meta
        meta[_KTK_HASH_BUCKET] = np.uint64(0)
        ddf = ddf.map_partitions(_hash_bucket, bucket_by, num_buckets, meta=meta)
        group_cols = [partition_on[0], _KTK_HASH_BUCKET]
    else:
        group_cols = [partition_on[0]]

    ddf = ddf.groupby(by=group_cols)
    ddf = ddf.apply(
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
    return ddf


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
    mps = map_delayed(input_to_mps, delayed_tasks)

    if sort_partitions_by:
        mps = map_delayed(
            partial(
                MetaPartition.apply,
                func=partial(sort_values_categorical, column=sort_partitions_by),
            ),
            mps,
        )
    if partition_on:
        mps = map_delayed(MetaPartition.partition_on, mps, partition_on=partition_on)
    if secondary_indices:
        mps = map_delayed(MetaPartition.build_indices, mps, columns=secondary_indices)

    return map_delayed(
        MetaPartition.store_dataframes,
        mps,
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
    if _KTK_HASH_BUCKET in df:
        df = df.drop(_KTK_HASH_BUCKET, axis=1)
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
