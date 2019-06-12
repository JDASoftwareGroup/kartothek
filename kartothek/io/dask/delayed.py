# -*- coding: utf-8 -*-


from collections import defaultdict
from functools import partial

import dask
from dask import delayed

from kartothek.core import naming
from kartothek.core.factory import _ensure_factory
from kartothek.core.naming import DEFAULT_METADATA_VERSION
from kartothek.core.utils import _check_callable
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.delete import (
    delete_common_metadata,
    delete_indices,
    delete_top_level_metadata,
)
from kartothek.io_components.docs import default_docs
from kartothek.io_components.gc import delete_files, dispatch_files_to_gc
from kartothek.io_components.merge import align_datasets
from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import (
    _ensure_compatible_indices,
    normalize_arg,
    normalize_args,
    validate_partition_keys,
)
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)

from ._update import _update_dask_partitions_one_to_one
from ._utils import _maybe_get_categoricals_from_index, map_delayed


def _delete_all_additional_metadata(dataset_factory):
    delete_indices(dataset_factory=dataset_factory)
    delete_common_metadata(dataset_factory=dataset_factory)
    # This is just a dummy return to chain
    return dataset_factory.dataset_uuid


def _delete_tl_metadata(dataset_factory, *args):
    """
    This function serves as a collector function for delayed objects. Therefore
    allowing additional arguments which are not used.
    """
    delete_top_level_metadata(dataset_factory=dataset_factory)


@default_docs
def delete_dataset__delayed(dataset_uuid=None, store=None, factory=None):
    """
    Parameters
    ----------
    """
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_schema=False,
        load_dataset_metadata=False,
    )

    gc = garbage_collect_dataset__delayed(
        dataset_uuid=None, store=None, chunk_size=100, factory=dataset_factory
    )

    mps = dispatch_metapartitions_from_factory(dataset_factory)

    delayed_dataset_uuid = delayed(_delete_all_additional_metadata)(
        dataset_factory=dataset_factory
    )

    mps = map_delayed(
        mps,
        MetaPartition.delete_from_store,
        store=store,
        dataset_uuid=delayed_dataset_uuid,
    )

    return delayed(_delete_tl_metadata)(dataset_factory, mps, gc)


def garbage_collect_dataset__delayed(
    dataset_uuid=None, store=None, chunk_size=100, factory=None
):
    """
    Remove auxiliary files that are no longer tracked by the dataset.

    These files include indices that are no longer referenced by the metadata
    as well as files in the directories of the tables that are no longer
    referenced. The latter is only applied to static datasets.

    Parameters
    ----------
    dataset_uuid: basestring
        The UUID of the dataset to be deleted
    store: callable
        A function returning a KeyValueStore.
    chunk_size: int
        Number of files that should be deleted in a single job.

    Returns
    -------
    tasks: list of dask.delayed
    """
    nested_files = dispatch_files_to_gc(
        dataset_uuid=dataset_uuid,
        store_factory=store,
        chunk_size=chunk_size,
        factory=factory,
    )
    return [delayed(delete_files)(files, store_factory=store) for files in nested_files]


def _load_and_merge_mps(mp_list, store, label_merger, metadata_merger, merge_tasks):
    mp_list = [mp.load_dataframes(store=store) for mp in mp_list]
    mp = MetaPartition.merge_metapartitions(
        mp_list, label_merger=label_merger, metadata_merger=metadata_merger
    )
    mp = mp.concat_dataframes()

    for task in merge_tasks:
        mp = mp.merge_dataframes(**task)

    return mp


@default_docs
def merge_datasets_as_delayed(
    left_dataset_uuid,
    right_dataset_uuid,
    store,
    merge_tasks,
    match_how="exact",
    label_merger=None,
    metadata_merger=None,
):
    """
    A dask.delayed graph to perform the merge of two full kartothek datasets.

    Parameters
    ----------
    left_dataset_uuid : basestring
        UUID for left dataset (order does not matter in all merge schemas)
    right_dataset_uuid : basestring
        UUID for right dataset (order does not matter in all merge schemas)
    match_how : basestring or callable, {left, right, prefix, exact}
        Define the partition label matching scheme.
        Available implementations are:

    Parameters
    ----------
    left_dataset_uuid : str
        UUID for left dataset (order does not matter in all merge schemas)
    right_dataset_uuid : str
        UUID for right dataset (order does not matter in all merge schemas)
    match_how : Union[str, Callable]
        Define the partition label matching scheme.
        Available implementations are:

        * left (right) : The left (right) partitions are considered to be
                            the base partitions and **all** partitions of the
                            right (left) dataset are joined to the left
                            partition. This should only be used if one of the
                            datasets contain very few partitions.
        * prefix : The labels of the partitions of the dataset with fewer
                    partitions are considered to be the prefixes to the
                    right dataset
        * exact : All partition labels of the left dataset need to have
                    an exact match in the right dataset
        * callable : A callable with signature func(left, right) which
                        returns a boolean to determine if the partitions match

        If True, an exact match of partition labels between the to-be-merged
        datasets is required in order to merge.
        If False (Default), the partition labels of the dataset with fewer
        partitions are interpreted as prefixes.
    merge_tasks : List[Dict]
        A list of merge tasks. Each item in this list is a dictionary giving
        explicit instructions for a specific merge.
        Each dict should contain key/values:

        * `left`: The table for the left dataframe
        * `right`: The table for the right dataframe
        * 'output_label' : The table for the merged dataframe
        * `merge_func`: A callable with signature
                        `merge_func(left_df, right_df, merge_kwargs)` to
                        handle the data preprocessing and merging.
                        Default pandas.merge
        * 'merge_kwargs' : The kwargs to be passed to the `merge_func`

        Example:

        .. code::

            >>> merge_tasks = [
            ...     {
            ...         "left": "left_dict",
            ...         "right": "right_dict",
            ...         "merge_kwargs": {"kwargs of merge_func": ''},
            ...         "output_label": 'merged_core_data'
            ...     },
            ... ]

    """
    _check_callable(store)

    mps = align_datasets(
        left_dataset_uuid=left_dataset_uuid,
        right_dataset_uuid=right_dataset_uuid,
        store=store,
        match_how=match_how,
    )
    mps = map_delayed(
        mps,
        _load_and_merge_mps,
        store=store,
        label_merger=label_merger,
        metadata_merger=metadata_merger,
        merge_tasks=merge_tasks,
    )

    return mps


def _load_and_concat_metapartitions_inner(mps, args, kwargs):
    return MetaPartition.concat_metapartitions(
        [mp.load_dataframes(*args, **kwargs) for mp in mps]
    )


def _load_and_concat_metapartitions(list_of_mps, *args, **kwargs):
    return [
        delayed(_load_and_concat_metapartitions_inner)(mps, args, kwargs)
        for mps in list_of_mps
    ]


def _identity():
    def _id(x):
        return x

    return _id


@default_docs
def read_dataset_as_delayed_metapartitions(
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
):
    """
    A collection of dask.delayed objects to retrieve a dataset from store where each
    partition is loaded as a :class:`~kartothek.io_components.metapartition.MetaPartition`.

    .. seealso:

        :func:`~kartothek.io.dask.read_dataset_as_delayed`

    Parameters
    ----------

    """
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=load_dataset_metadata,
    )
    store = ds_factory.store_factory
    mps = dispatch_metapartitions_from_factory(
        dataset_factory=ds_factory,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        label_filter=label_filter,
        predicates=predicates,
    )

    if concat_partitions_on_primary_index:
        mps = _load_and_concat_metapartitions(
            mps,
            store=store,
            tables=tables,
            columns=columns,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )
    else:
        mps = map_delayed(
            mps,
            MetaPartition.load_dataframes,
            store=store,
            tables=tables,
            columns=columns,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )

    categoricals_from_index = _maybe_get_categoricals_from_index(
        ds_factory, categoricals
    )

    if categoricals_from_index:
        func_dict = defaultdict(_identity)
        func_dict.update(
            {
                table: partial(_cast_categorical_to_index_cat, categories=cats)
                for table, cats in categoricals_from_index.items()
            }
        )
        mps = map_delayed(mps, MetaPartition.apply, func_dict, type_safe=True)

    return mps


def _get_data(mp, table=None):
    """
    Task to avoid serialization of lambdas
    """
    if table:
        return mp.data[table]
    else:
        return mp.data


@default_docs
def read_dataset_as_delayed(
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
):
    """
    A collection of dask.delayed objects to retrieve a dataset from store
    where each partition is loaded as a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    """
    mps = read_dataset_as_delayed_metapartitions(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        tables=tables,
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        load_dataset_metadata=False,
        predicates=predicates,
    )
    return map_delayed(mps, _get_data)


@default_docs
def read_table_as_delayed(
    dataset_uuid=None,
    store=None,
    table=None,
    columns=None,
    concat_partitions_on_primary_index=False,
    predicate_pushdown_to_io=True,
    categoricals=None,
    label_filter=None,
    dates_as_object=False,
    predicates=None,
    factory=None,
):
    """
    A collection of dask.delayed objects to retrieve a single table from
    a dataset as partition-individual :class:`~pandas.DataFrame` instances.

    You can transform the collection of ``dask.delayed`` objects into
    a ``dask.dataframe`` using the following code snippet. As older kartothek
    specifications don't store schema information, this must be provided by
    a separate code path.

    .. code ::

        >>> import dask.dataframe as dd
        >>> ddf_tasks = read_table_as_delayed(…)
        >>> meta = …
        >>> ddf = dd.from_delayed(ddf_tasks, meta=meta)

    Parameters
    ----------
    """
    if table is None:
        raise TypeError("Parameter `table` is not optional.")
    if not isinstance(columns, dict):
        columns = {table: columns}
    mps = read_dataset_as_delayed_metapartitions(
        dataset_uuid=dataset_uuid,
        store=store,
        tables=[table],
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        load_dataset_metadata=False,
        predicates=predicates,
        factory=factory,
    )
    return map_delayed(mps, partial(_get_data, table=table))


def _cast_categorical_to_index_cat(df, categories):
    try:
        return df.astype(categories)
    except ValueError as verr:
        # Should be fixed by pandas>=0.24.0
        if "buffer source array is read-only" in str(verr):
            new_cols = {}
            for cat in categories:
                new_cols[cat] = df[cat].astype(df[cat].dtype.categories.dtype)
                new_cols[cat] = new_cols[cat].astype(categories[cat])
            return df.assign(**new_cols)
        raise


@default_docs
def update_dataset_from_delayed(
    delayed_tasks,
    store=None,
    dataset_uuid=None,
    delete_scope=None,
    metadata=None,
    df_serializer=None,
    metadata_merger=None,
    default_metadata_version=DEFAULT_METADATA_VERSION,
    partition_on=None,
    sort_partitions_by=None,
    secondary_indices=None,
    factory=None,
):
    """
    A dask.delayed graph to add and store a list of dictionaries containing
    dataframes to a kartothek dataset in store. The input should be a list
    (or splitter pipeline) containing
    :class:`~karothek.io.metapartition.MetaPartition`. If you want to use this
    pipeline step for just deleting partitions without adding new ones you
    have to give an empty meta partition as input (``[Metapartition(None)]``).

    Parameters
    ----------
    """
    partition_on = normalize_arg("partition_on", partition_on)
    secondary_indices = normalize_arg("secondary_indices", secondary_indices)
    delete_scope = dask.delayed(normalize_arg)("delete_scope", delete_scope)

    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
        ds_factory=factory,
    )

    secondary_indices = _ensure_compatible_indices(ds_factory, secondary_indices)
    mps = _update_dask_partitions_one_to_one(
        delayed_tasks=delayed_tasks,
        secondary_indices=secondary_indices,
        metadata_version=metadata_version,
        partition_on=partition_on,
        store_factory=store,
        df_serializer=df_serializer,
        dataset_uuid=dataset_uuid,
        sort_partitions_by=sort_partitions_by,
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
def store_delayed_as_dataset(
    delayed_tasks,
    store,
    dataset_uuid=None,
    metadata=None,
    df_serializer=None,
    overwrite=False,
    metadata_merger=None,
    metadata_version=naming.DEFAULT_METADATA_VERSION,
    partition_on=None,
    metadata_storage_format=naming.DEFAULT_METADATA_STORAGE_FORMAT,
    secondary_indices=None,
):
    """
    Transform and store a list of dictionaries containing
    dataframes to a kartothek dataset in store.

    Parameters
    ----------
    delayed_tasks: list of dask.delayed
        Every delayed object represents a partition and should be accepted by
        :func:`~kartothek.io_components.metapartition.parse_input_to_metapartition`


    Returns
    -------
    A dask.delayed dataset object.
    """
    _check_callable(store)
    if dataset_uuid is None:
        dataset_uuid = gen_uuid()

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    input_to_mps = partial(
        parse_input_to_metapartition, metadata_version=metadata_version
    )
    mps = map_delayed(delayed_tasks, input_to_mps)

    if partition_on:
        mps = map_delayed(mps, MetaPartition.partition_on, partition_on=partition_on)

    if secondary_indices:
        mps = map_delayed(mps, MetaPartition.build_indices, columns=secondary_indices)

    mps = map_delayed(
        mps,
        MetaPartition.store_dataframes,
        store=store,
        df_serializer=df_serializer,
        dataset_uuid=dataset_uuid,
    )

    return delayed(store_dataset_from_partitions)(
        mps,
        dataset_uuid=dataset_uuid,
        store=store,
        dataset_metadata=metadata,
        metadata_merger=metadata_merger,
        metadata_storage_format=metadata_storage_format,
    )
