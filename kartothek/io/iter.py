# -*- coding: utf-8 -*-

import warnings
from functools import partial
from typing import cast

from kartothek.core.docs import default_docs
from kartothek.core.factory import _ensure_factory
from kartothek.core.naming import (
    DEFAULT_METADATA_STORAGE_FORMAT,
    DEFAULT_METADATA_VERSION,
)
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import (
    _ensure_compatible_indices,
    normalize_args,
    raise_if_indices_overlap,
    sort_values_categorical,
    validate_partition_keys,
)
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)


@default_docs
def read_dataset_as_metapartitions__iterator(
    dataset_uuid=None,
    store=None,
    tables=None,
    columns=None,
    concat_partitions_on_primary_index=False,
    predicate_pushdown_to_io=True,
    categoricals=None,
    label_filter=None,
    dates_as_object=False,
    load_dataset_metadata=False,
    predicates=None,
    factory=None,
    dispatch_by=None,
    dispatch_metadata=True,
):
    """

    A Python iterator to retrieve a dataset from store where each
    partition is loaded as a :class:`~kartothek.io_components.metapartition.MetaPartition`.

    .. seealso:

        :func:`~kartothek.io_components.read.read_dataset_as_dataframes__iterator`

    Parameters
    ----------

    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=load_dataset_metadata,
    )
    store = ds_factory.store
    mps = dispatch_metapartitions_from_factory(
        ds_factory,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        label_filter=label_filter,
        predicates=predicates,
        dispatch_by=dispatch_by,
        dispatch_metadata=dispatch_metadata,
    )

    for mp in mps:
        if concat_partitions_on_primary_index or dispatch_by:
            mp = MetaPartition.concat_metapartitions(
                [
                    mp_inner.load_dataframes(
                        store=store,
                        tables=tables,
                        columns=columns,
                        categoricals=categoricals,
                        predicate_pushdown_to_io=predicate_pushdown_to_io,
                        predicates=predicates,
                    )
                    for mp_inner in mp
                ]
            )
        else:
            mp = cast(MetaPartition, mp)
            mp = mp.load_dataframes(
                store=store,
                tables=tables,
                columns=columns,
                categoricals=categoricals,
                predicate_pushdown_to_io=predicate_pushdown_to_io,
                dates_as_object=dates_as_object,
                predicates=predicates,
            )
        yield mp


@default_docs
def read_dataset_as_dataframes__iterator(
    dataset_uuid=None,
    store=None,
    tables=None,
    columns=None,
    concat_partitions_on_primary_index=False,
    predicate_pushdown_to_io=True,
    categoricals=None,
    label_filter=None,
    dates_as_object=False,
    predicates=None,
    factory=None,
    dispatch_by=None,
):
    """
    A Python iterator to retrieve a dataset from store where each
    partition is loaded as a :class:`~pandas.DataFrame`.

    Parameters
    ----------

    Returns
    -------
    list
        A list containing a dictionary for each partition. The dictionaries
        keys are the in-partition file labels and the values are the
        corresponding dataframes.

    Examples
    --------
    Dataset in store contains two partitions with two files each

    .. code ::

        >>> import storefact
        >>> from kartothek.io.iter import read_dataset_as_dataframes__iterator

        >>> store = storefact.get_store_from_url('s3://bucket_with_dataset')

        >>> dataframes = read_dataset_as_dataframes__iterator('dataset_uuid', store)
        >>> next(dataframes)
        [
            # First partition
            {'core': pd.DataFrame, 'lookup': pd.DataFrame}
        ]

        >>> next(dataframes)
        [
            # Second partition
            {'core': pd.DataFrame, 'lookup': pd.DataFrame}
        ]
    """
    mp_iter = read_dataset_as_metapartitions__iterator(
        dataset_uuid=dataset_uuid,
        store=store,
        tables=tables,
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        load_dataset_metadata=False,
        predicates=predicates,
        factory=factory,
        dispatch_by=dispatch_by,
        dispatch_metadata=False,
    )
    for mp in mp_iter:
        yield mp.data


@default_docs
@normalize_args
def update_dataset_from_dataframes__iter(
    df_generator,
    store=None,
    dataset_uuid=None,
    delete_scope=None,
    metadata=None,
    df_serializer=None,
    metadata_merger=None,
    central_partition_metadata=True,
    default_metadata_version=DEFAULT_METADATA_VERSION,
    partition_on=None,
    load_dynamic_metadata=True,
    sort_partitions_by=None,
    secondary_indices=None,
    factory=None,
):
    """
    Update a kartothek dataset in store iteratively, using a generator of dataframes.

    Useful for datasets which do not fit into memory.

    Parameters
    ----------

    Returns
    -------
    The dataset metadata object (:class:`~kartothek.core.dataset.DatasetMetadata`).
    """
    if load_dynamic_metadata is not True:
        warnings.warn(
            "The keyword `load_dynamic_metadata` has no use and will be removed soon",
            DeprecationWarning,
        )

    if central_partition_metadata is not True:
        warnings.warn(
            "The keyword `central_partition_metadata` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )
    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        ds_factory=factory,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
    )

    secondary_indices = _ensure_compatible_indices(ds_factory, secondary_indices)

    if sort_partitions_by:  # Define function which sorts each partition by column
        sort_partitions_by_fn = partial(
            sort_values_categorical, column=sort_partitions_by
        )

    new_partitions = []
    for df in df_generator:
        mp = parse_input_to_metapartition(
            df,
            metadata_version=metadata_version,
            expected_secondary_indices=secondary_indices,
        )

        if sort_partitions_by:
            mp = mp.apply(sort_partitions_by_fn)

        if partition_on:
            mp = mp.partition_on(partition_on=partition_on)

        if secondary_indices:
            mp = mp.build_indices(columns=secondary_indices)

        # Store dataframe, thereby clearing up the dataframe from the `mp` metapartition
        mp = mp.store_dataframes(
            store=store, df_serializer=df_serializer, dataset_uuid=dataset_uuid
        )

        new_partitions.append(mp)

    return update_dataset_from_partitions(
        new_partitions,
        store_factory=store,
        dataset_uuid=dataset_uuid,
        ds_factory=ds_factory,
        delete_scope=delete_scope,
        metadata=metadata,
        metadata_merger=metadata_merger,
    )


@default_docs
@normalize_args
def store_dataframes_as_dataset__iter(
    df_generator,
    store,
    dataset_uuid=None,
    metadata=None,
    partition_on=None,
    df_serializer=None,
    overwrite=False,
    metadata_storage_format=DEFAULT_METADATA_STORAGE_FORMAT,
    metadata_version=DEFAULT_METADATA_VERSION,
    secondary_indices=None,
):
    """
    Store `pd.DataFrame` s iteratively as a partitioned dataset with multiple tables (files).

    Useful for datasets which do not fit into memory.

    Parameters
    ----------

    Returns
    -------
    dataset: kartothek.core.dataset.DatasetMetadata
        The stored dataset.

    """

    if dataset_uuid is None:
        dataset_uuid = gen_uuid()

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    raise_if_indices_overlap(partition_on, secondary_indices)

    new_partitions = []
    for df in df_generator:
        mp = parse_input_to_metapartition(df, metadata_version=metadata_version)

        if partition_on:
            mp = mp.partition_on(partition_on)

        if secondary_indices:
            mp = mp.build_indices(secondary_indices)

        # Store dataframe, thereby clearing up the dataframe from the `mp` metapartition
        mp = mp.store_dataframes(
            store=store, dataset_uuid=dataset_uuid, df_serializer=df_serializer
        )

        # Add `kartothek.io_components.metapartition.MetaPartition` object to list to track partitions
        new_partitions.append(mp)

    # Store metadata and return `kartothek.DatasetMetadata` object
    return store_dataset_from_partitions(
        partition_list=new_partitions,
        dataset_uuid=dataset_uuid,
        store=store,
        dataset_metadata=metadata,
        metadata_storage_format=metadata_storage_format,
    )
