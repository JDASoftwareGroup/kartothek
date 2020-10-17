import random
import warnings
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    SupportsFloat,
    Union,
    cast,
)

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from simplekv import KeyValueStore

from kartothek.core.common_metadata import empty_dataframe_from_schema
from kartothek.core.docs import default_docs
from kartothek.core.factory import DatasetFactory, _ensure_factory
from kartothek.core.naming import DEFAULT_METADATA_VERSION
from kartothek.core.typing import StoreFactory, StoreInput
from kartothek.io.dask.compression import pack_payload, unpack_payload_pandas
from kartothek.io_components.metapartition import (
    _METADATA_SCHEMA,
    SINGLE_TABLE,
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import (
    _ensure_compatible_indices,
    check_single_table_dataset,
    normalize_arg,
    normalize_args,
    validate_partition_keys,
)
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
    write_partition,
)
from kartothek.serialization import DataFrameSerializer, PredicatesType

from ._shuffle import shuffle_store_dask_partitions
from ._utils import _maybe_get_categoricals_from_index
from .delayed import read_table_as_delayed


@default_docs
@normalize_args
def read_dataset_as_ddf(
    dataset_uuid=None,
    store=None,
    table=SINGLE_TABLE,
    columns=None,
    concat_partitions_on_primary_index=False,
    predicate_pushdown_to_io=True,
    categoricals=None,
    label_filter=None,
    dates_as_object=False,
    predicates=None,
    factory=None,
    dask_index_on=None,
    dispatch_by=None,
):
    """
    Retrieve a single table from a dataset as partition-individual :class:`~dask.dataframe.DataFrame` instance.

    Please take care when using categoricals with Dask. For index columns, this function will construct dataset
    wide categoricals. For all other columns, Dask will determine the categories on a partition level and will
    need to merge them when shuffling data.

    Parameters
    ----------
    dask_index_on: str
        Reconstruct (and set) a dask index on the provided index column. Cannot be used
        in conjunction with `dispatch_by`.

        For details on performance, see also `dispatch_by`
    """
    if dask_index_on is not None and not isinstance(dask_index_on, str):
        raise TypeError(
            f"The paramter `dask_index_on` must be a string but got {type(dask_index_on)}"
        )

    if dask_index_on is not None and dispatch_by is not None and len(dispatch_by) > 0:
        raise ValueError(
            "`read_dataset_as_ddf` got parameters `dask_index_on` and `dispatch_by`. "
            "Note that `dispatch_by` can only be used if `dask_index_on` is None."
        )

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    if len(ds_factory.tables) > 1:
        warnings.warn(
            "Trying to read a dataset with multiple internal tables. This functionality will be removed in the next "
            "major release. If you require a multi tabled data format, we recommend to switch to the kartothek Cube "
            "functionality. "
            "https://kartothek.readthedocs.io/en/stable/guide/cube/kartothek_cubes.html",
            DeprecationWarning,
        )

    if isinstance(columns, dict):
        columns = columns[table]
    meta = _get_dask_meta_for_dataset(
        ds_factory, table, columns, categoricals, dates_as_object
    )

    if columns is None:
        columns = list(meta.columns)

    # that we can use factories instead of dataset_uuids
    delayed_partitions = read_table_as_delayed(
        factory=ds_factory,
        table=table,
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals={table: categoricals},
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        predicates=predicates,
        dispatch_by=dask_index_on if dask_index_on else dispatch_by,
    )
    if dask_index_on:
        divisions = ds_factory.indices[dask_index_on].observed_values()
        divisions.sort()
        divisions = list(divisions)
        divisions.append(divisions[-1])
        return dd.from_delayed(
            delayed_partitions, meta=meta, divisions=divisions
        ).set_index(dask_index_on, divisions=divisions, sorted=True)
    else:
        return dd.from_delayed(delayed_partitions, meta=meta)


def _get_dask_meta_for_dataset(
    ds_factory, table, columns, categoricals, dates_as_object
):
    """
    Calculate a schema suitable for the dask dataframe meta from the dataset.
    """
    table_schema = ds_factory.table_meta[table]
    meta = empty_dataframe_from_schema(
        table_schema, columns=columns, date_as_object=dates_as_object
    )

    if categoricals:
        meta = meta.astype({col: "category" for col in categoricals})
        meta = dd.utils.clear_known_categories(meta, categoricals)

    categoricals_from_index = _maybe_get_categoricals_from_index(
        ds_factory, {table: categoricals}
    )
    if categoricals_from_index:
        meta = meta.astype(categoricals_from_index[table])
    return meta


def _shuffle_docs(func):
    func.__doc__ += """

    .. admonition:: Behavior without ``shuffle==False``

        In the case without ``partition_on`` every dask partition is mapped to a single kartothek partition

        In the case with ``partition_on`` every dask partition is mapped to N kartothek partitions, where N
        depends on the content of the respective partition, such that every resulting kartothek partition has
        only a single value in the respective ``partition_on`` columns.

    .. admonition:: Behavior with ``shuffle==True``

        ``partition_on`` is mandatory

        Perform a data shuffle to ensure that every primary key will have at most ``num_bucket``.

        .. note::
            The number of allowed buckets will have an impact on the required resources and runtime.
            Using a larger number of allowed buckets will usually reduce resource consumption and in some
            cases also improves runtime performance.

        :Example:

            >>> partition_on="primary_key"
            >>> num_buckets=2  # doctest: +SKIP
            primary_key=1/bucket1.parquet
            primary_key=1/bucket2.parquet

    .. note:: This can only be used for datasets with a single table!

    See also, :ref:`partitioning_dask`.

    Parameters
    ----------
    ddf: Union[dask.dataframe.DataFrame, None]
        The dask.Dataframe to be used to calculate the new partitions from. If this parameter is `None`, the update pipeline
        will only delete partitions without creating new ones.
    shuffle: bool
        If `True` and `partition_on` is requested, shuffle the data to reduce number of output partitions.

        See also, :ref:`shuffling`.

        .. warning::

            Dask uses a heuristic to determine how data is shuffled and there are two options, `partd` for local disk shuffling and `tasks` for distributed shuffling using a task graph. If there is no :class:`distributed.Client` in the context and the option is not set explicitly, dask will choose `partd` which may cause data loss when the graph is executed on a distributed cluster.

            Therefore, we recommend to specify the dask shuffle method explicitly, e.g. by using a context manager.

            .. code::

                with dask.config.set(shuffle='tasks'):
                    graph = update_dataset_from_ddf(...)
                graph.compute()

    repartition_ratio: Optional[Union[int, float]]
        If provided, repartition the dataframe before calculation starts to ``ceil(ddf.npartitions / repartition_ratio)``
    num_buckets: int
        If provided, the output partitioning will have ``num_buckets`` files per primary key partitioning.
        This effectively splits up the execution ``num_buckets`` times. Setting this parameter may be helpful when
        scaling.
        This only has an effect if ``shuffle==True``
    bucket_by:
        The subset of columns which should be considered for bucketing.

        This parameter ensures that groups of the given subset are never split
        across buckets within a given partition.

        Without specifying this the buckets will be created randomly.

        This only has an effect if ``shuffle==True``

        .. admonition:: Secondary indices

            This parameter has a strong effect on the performance of secondary
            indices. Since it guarantees that a given tuple of the subset will
            be entirely put into the same file you can build efficient indices
            with this approach.

        .. note::

            Only columns with data types which can be hashed are allowed to be used in this.
"""
    return func


@default_docs
@_shuffle_docs
def store_dataset_from_ddf(
    ddf: dd.DataFrame,
    store: StoreInput,
    dataset_uuid: str,
    table: str = SINGLE_TABLE,
    secondary_indices: Optional[List[str]] = None,
    shuffle: bool = False,
    repartition_ratio: Optional[SupportsFloat] = None,
    num_buckets: int = 1,
    sort_partitions_by: Optional[Union[List[str], str]] = None,
    delete_scope: Optional[Iterable[Mapping[str, str]]] = None,
    metadata: Optional[Mapping] = None,
    df_serializer: Optional[DataFrameSerializer] = None,
    metadata_merger: Optional[Callable] = None,
    metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: Optional[List[str]] = None,
    bucket_by: Optional[Union[List[str], str]] = None,
    overwrite: bool = False,
):
    """
    Store a dataset from a dask.dataframe.
    """
    partition_on = normalize_arg("partition_on", partition_on)
    secondary_indices = normalize_arg("secondary_indices", secondary_indices)
    sort_partitions_by = normalize_arg("sort_partitions_by", sort_partitions_by)
    bucket_by = normalize_arg("bucket_by", bucket_by)
    store = normalize_arg("store", store)
    delete_scope = dask.delayed(normalize_arg)("delete_scope", delete_scope)

    if table is None:
        raise TypeError("The parameter `table` is not optional.")

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=None, load_dataset_metadata=True
    )

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)
    mps = _write_dataframe_partitions(
        ddf=ddf,
        store=store,
        dataset_uuid=dataset_uuid,
        table=table,
        secondary_indices=secondary_indices,
        shuffle=shuffle,
        repartition_ratio=repartition_ratio,
        num_buckets=num_buckets,
        sort_partitions_by=sort_partitions_by,
        df_serializer=df_serializer,
        metadata_version=metadata_version,
        partition_on=partition_on,
        bucket_by=bucket_by,
    )
    return dask.delayed(store_dataset_from_partitions)(
        mps,
        store=ds_factory.store_factory if ds_factory else store,
        dataset_uuid=ds_factory.dataset_uuid if ds_factory else dataset_uuid,
        dataset_metadata=metadata,
        metadata_merger=metadata_merger,
    )


def _write_dataframe_partitions(
    ddf: dd.DataFrame,
    store: StoreFactory,
    dataset_uuid: str,
    table: str,
    secondary_indices: List[str],
    shuffle: bool,
    repartition_ratio: Optional[SupportsFloat],
    num_buckets: int,
    sort_partitions_by: List[str],
    df_serializer: Optional[DataFrameSerializer],
    metadata_version: int,
    partition_on: List[str],
    bucket_by: List[str],
) -> dd.Series:
    if repartition_ratio and ddf is not None:
        ddf = ddf.repartition(
            npartitions=int(np.ceil(ddf.npartitions / repartition_ratio))
        )

    if ddf is None:
        mps = dd.from_pandas(
            pd.Series(
                [parse_input_to_metapartition(None, metadata_version=metadata_version)]
            ),
            npartitions=1,
        )
    else:
        if shuffle:
            mps = shuffle_store_dask_partitions(
                ddf=ddf,
                table=table,
                secondary_indices=secondary_indices,
                metadata_version=metadata_version,
                partition_on=partition_on,
                store_factory=store,
                df_serializer=df_serializer,
                dataset_uuid=dataset_uuid,
                num_buckets=num_buckets,
                sort_partitions_by=sort_partitions_by,
                bucket_by=bucket_by,
            )
        else:
            mps = ddf.map_partitions(
                write_partition,
                secondary_indices=secondary_indices,
                metadata_version=metadata_version,
                partition_on=partition_on,
                store_factory=store,
                df_serializer=df_serializer,
                dataset_uuid=dataset_uuid,
                sort_partitions_by=sort_partitions_by,
                dataset_table_name=table,
                meta=(MetaPartition),
            )
    return mps


@default_docs
@_shuffle_docs
def update_dataset_from_ddf(
    ddf: dd.DataFrame,
    store: Optional[StoreInput] = None,
    dataset_uuid: Optional[str] = None,
    table: str = SINGLE_TABLE,
    secondary_indices: Optional[List[str]] = None,
    shuffle: bool = False,
    repartition_ratio: Optional[SupportsFloat] = None,
    num_buckets: int = 1,
    sort_partitions_by: Optional[Union[List[str], str]] = None,
    delete_scope: Optional[Iterable[Mapping[str, str]]] = None,
    metadata: Optional[Mapping] = None,
    df_serializer: Optional[DataFrameSerializer] = None,
    metadata_merger: Optional[Callable] = None,
    default_metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: Optional[List[str]] = None,
    factory: Optional[DatasetFactory] = None,
    bucket_by: Optional[Union[List[str], str]] = None,
):
    """
    Update a dataset from a dask.dataframe.
    """
    partition_on = normalize_arg("partition_on", partition_on)
    secondary_indices = normalize_arg("secondary_indices", secondary_indices)
    sort_partitions_by = normalize_arg("sort_partitions_by", sort_partitions_by)
    bucket_by = normalize_arg("bucket_by", bucket_by)
    store = normalize_arg("store", store)
    delete_scope = dask.delayed(normalize_arg)("delete_scope", delete_scope)

    if table is None:
        raise TypeError("The parameter `table` is not optional.")
    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
        ds_factory=factory,
    )

    _ensure_compatible_indices(ds_factory, secondary_indices)

    if ds_factory is not None:
        check_single_table_dataset(ds_factory, table)

    mps = _write_dataframe_partitions(
        ddf=ddf,
        store=store,
        dataset_uuid=dataset_uuid or ds_factory.dataset_uuid,
        table=table,
        secondary_indices=secondary_indices,
        shuffle=shuffle,
        repartition_ratio=repartition_ratio,
        num_buckets=num_buckets,
        sort_partitions_by=sort_partitions_by,
        df_serializer=df_serializer,
        metadata_version=metadata_version,
        partition_on=cast(List[str], partition_on),
        bucket_by=bucket_by,
    )
    return dask.delayed(update_dataset_from_partitions)(
        mps,
        store_factory=store,
        dataset_uuid=dataset_uuid,
        ds_factory=ds_factory,
        delete_scope=delete_scope,
        metadata=metadata,
        metadata_merger=metadata_merger,
    )


@default_docs
@normalize_args
def collect_dataset_metadata(
    store: Optional[Callable[[], KeyValueStore]] = None,
    dataset_uuid: Optional[str] = None,
    table_name: str = SINGLE_TABLE,
    predicates: Optional[PredicatesType] = None,
    frac: float = 1.0,
    factory: Optional[DatasetFactory] = None,
) -> dd.DataFrame:
    """
    Collect parquet metadata of the dataset. The `frac` parameter can be used to select a subset of the data.

    .. warning::
      If the size of the partitions is not evenly distributed, e.g. some partitions might be larger than others,
      the metadata returned is not a good approximation for the whole dataset metadata.
    .. warning::
      Using the `frac` parameter is not encouraged for a small number of total partitions.


    Parameters
    ----------
    predicates
      Kartothek predicates to apply filters on the data for which to gather statistics

      .. warning::
          Filtering will only be applied for predicates on indices.
          The evaluation of the predicates therefore will therefore only return an approximate result.

    frac
      Fraction of the total number of partitions to use for gathering statistics. `frac == 1.0` will use all partitions.

    Returns
    -------
    A dask.DataFrame containing the following information about dataset statistics:
       * `partition_label`: File name of the parquet file, unique to each physical partition.
       * `row_group_id`: Index of the row groups within one parquet file.
       * `row_group_compressed_size`: Byte size of the data within one row group.
       * `row_group_uncompressed_size`: Byte size (uncompressed) of the data within one row group.
       * `number_rows_total`: Total number of rows in one parquet file.
       * `number_row_groups`: Number of row groups in one parquet file.
       * `serialized_size`: Serialized size of the parquet file.
       * `number_rows_per_row_group`: Number of rows per row group.

    Raises
    ------
    ValueError
      If no metadata could be retrieved, raise an error.

    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(
            f"Invalid value for parameter `frac`: {frac}."
            "Please make sure to provide a value larger than 0.0 and smaller than or equal to 1.0 ."
        )
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    mps = list(
        dispatch_metapartitions_from_factory(dataset_factory, predicates=predicates)
    )
    if mps:
        random.shuffle(mps)
        # ensure that even with sampling at least one metapartition is returned
        cutoff_index = max(1, int(len(mps) * frac))
        mps = mps[:cutoff_index]
        ddf = dd.from_delayed(
            [
                dask.delayed(MetaPartition.get_parquet_metadata)(
                    mp, store=dataset_factory.store_factory, table_name=table_name
                )
                for mp in mps
            ],
            meta=_METADATA_SCHEMA,
        )
    else:
        df = pd.DataFrame(columns=_METADATA_SCHEMA.keys())
        df = df.astype(_METADATA_SCHEMA)
        ddf = dd.from_pandas(df, npartitions=1)

    return ddf


def _unpack_hash(df, unpack_meta, subset):
    df = unpack_payload_pandas(df, unpack_meta)
    if subset:
        df = df[subset]
    return _hash_partition(df)


def _hash_partition(part):
    return pd.util.hash_pandas_object(part, index=False).sum()


@default_docs
@normalize_args
def hash_dataset(
    store: Optional[StoreInput] = None,
    dataset_uuid: Optional[str] = None,
    subset=None,
    group_key=None,
    table: str = SINGLE_TABLE,
    predicates: Optional[PredicatesType] = None,
    factory: Optional[DatasetFactory] = None,
) -> dd.Series:
    """
    Calculate a partition wise, or group wise, hash of the dataset.

    .. note::

        We do not guarantee the hash values to remain constant accross versions.


    Example output::

        Assuming a dataset with two unique values in column `P` this gives

        >>> hash_dataset(factory=dataset_with_index_factory,group_key=["P"]).compute()
        ... P
        ... 1    11462879952839863487
        ... 2    12568779102514529673
        ... dtype: uint64

    Parameters
    ----------
    subset:
        If provided, only take these columns into account when hashing the dataset
    group_key:
        If provided, calculate hash per group instead of per partition
    """
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    columns = subset
    if subset and group_key:
        columns = sorted(set(subset) | set(group_key))
    ddf = read_dataset_as_ddf(
        table=table,
        predicates=predicates,
        factory=dataset_factory,
        columns=columns,
        dates_as_object=True,
    )
    if not group_key:
        return ddf.map_partitions(_hash_partition, meta="uint64").astype("uint64")
    else:
        ddf2 = pack_payload(ddf, group_key=group_key)
        return (
            ddf2.groupby(group_key)
            .apply(_unpack_hash, unpack_meta=ddf._meta, subset=subset, meta="uint64")
            .astype("uint64")
        )
