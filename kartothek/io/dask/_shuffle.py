from functools import partial
from typing import List, Optional, Sequence, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from kartothek.core.typing import StoreFactory
from kartothek.io.dask.compression import pack_payload, unpack_payload_pandas
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import write_partition
from kartothek.serialization import DataFrameSerializer

_KTK_HASH_BUCKET = "__KTK_HASH_BUCKET"


def _hash_bucket(df: pd.DataFrame, subset: Optional[Sequence[str]], num_buckets: int):
    """
    Categorize each row of `df` based on the data in the columns `subset`
    into `num_buckets` values. This is based on `pandas.util.hash_pandas_object`
    """

    if not subset:
        subset = df.columns
    hash_arr = pd.util.hash_pandas_object(df[subset], index=False)
    buckets = hash_arr % num_buckets

    available_bit_widths = np.array([8, 16, 32, 64])
    mask = available_bit_widths > np.log2(num_buckets)
    bit_width = min(available_bit_widths[mask])
    return df.assign(**{_KTK_HASH_BUCKET: buckets.astype(f"uint{bit_width}")})


def shuffle_store_dask_partitions(
    ddf: dd.DataFrame,
    table: str,
    secondary_indices: Optional[Union[str, Sequence[str]]],
    metadata_version: int,
    partition_on: List[str],
    store_factory: StoreFactory,
    df_serializer: Optional[DataFrameSerializer],
    dataset_uuid: str,
    num_buckets: int,
    sort_partitions_by: List[str],
    bucket_by: Sequence[str],
) -> da.Array:
    """
    Perform a dataset update with dask reshuffling to control partitioning.

    The shuffle operation will perform the following steps

    1. Pack payload data

       Payload data is serialized and compressed into a single byte value using
       ``distributed.protocol.serialize_bytes``, see also ``pack_payload``.

    2. Apply bucketing

       Hash the column subset ``bucket_by`` and distribute the hashes in
       ``num_buckets`` bins/buckets. Internally every bucket is identified by an
       integer and we will create one physical file for every bucket ID. The
       bucket ID is not exposed to the user and is dropped after the shuffle,
       before the store. This is done since we do not want to guarantee at the
       moment, that the hash function remains stable.

    3. Perform shuffle (dask.DataFrame.groupby.apply)

        The groupby key will be the combination of ``partition_on`` fields and the
        hash bucket ID. This will create a physical file for every unique tuple
        in ``partition_on + bucket_ID``. The function which is applied to the
        dataframe will perform all necessary subtask for storage of the dataset
        (partition_on, index calc, etc.).

    4. Unpack data (within the apply-function)

        After the shuffle, the first step is to unpack the payload data since
        the follow up tasks will require the full dataframe.

    5. Pre storage processing and parquet serialization

        We apply important pre storage processing like sorting data, applying
        final partitioning (at this time there should be only one group in the
        payload data but using the ``MetaPartition.partition_on`` guarantees the
        appropriate data structures kartothek expects are created.).
        After the preprocessing is done, the data is serialized and stored as
        parquet. The applied function will return an (empty) MetaPartition with
        indices and metadata which will then be used to commit the dataset.

    Returns
    -------

    A dask.Array holding relevant MetaPartition objects as values

    """
    if ddf.npartitions == 0:
        return ddf

    group_cols = partition_on.copy()

    if num_buckets is None:
        raise ValueError("``num_buckets`` must not be None when shuffling data.")

    meta = ddf._meta
    meta[_KTK_HASH_BUCKET] = np.uint64(0)
    ddf = ddf.map_partitions(_hash_bucket, bucket_by, num_buckets, meta=meta)
    group_cols.append(_KTK_HASH_BUCKET)

    unpacked_meta = ddf._meta

    ddf = pack_payload(ddf, group_key=group_cols)
    ddf = ddf.groupby(by=group_cols)
    ddf = ddf.apply(
        partial(
            _unpack_store_partition,
            secondary_indices=secondary_indices,
            sort_partitions_by=sort_partitions_by,
            table=table,
            dataset_uuid=dataset_uuid,
            partition_on=partition_on,
            store_factory=store_factory,
            df_serializer=df_serializer,
            metadata_version=metadata_version,
            unpacked_meta=unpacked_meta,
        ),
        meta=("MetaPartition", "object"),
    )
    return ddf


def _unpack_store_partition(
    df: pd.DataFrame,
    secondary_indices: List[str],
    sort_partitions_by: List[str],
    table: str,
    dataset_uuid: str,
    partition_on: Optional[List[str]],
    store_factory: StoreFactory,
    df_serializer: DataFrameSerializer,
    metadata_version: int,
    unpacked_meta: pd.DataFrame,
) -> MetaPartition:
    """Unpack payload data and store partition"""
    df = unpack_payload_pandas(df, unpacked_meta)
    if _KTK_HASH_BUCKET in df:
        df = df.drop(_KTK_HASH_BUCKET, axis=1)
    return write_partition(
        partition_df=df,
        secondary_indices=secondary_indices,
        sort_partitions_by=sort_partitions_by,
        dataset_table_name=table,
        dataset_uuid=dataset_uuid,
        partition_on=partition_on,
        store_factory=store_factory,
        df_serializer=df_serializer,
        metadata_version=metadata_version,
    )
