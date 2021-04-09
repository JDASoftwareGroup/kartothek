from functools import partial
from typing import List, Optional, Sequence

import dask
from dask import delayed
from dask.delayed import Delayed

from kartothek.core import naming
from kartothek.core.docs import default_docs
from kartothek.core.factory import _ensure_factory
from kartothek.core.naming import DEFAULT_METADATA_VERSION
from kartothek.core.typing import StoreInput
from kartothek.core.utils import lazy_store
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.delete import (
    delete_common_metadata,
    delete_indices,
    delete_top_level_metadata,
)
from kartothek.io_components.gc import delete_files, dispatch_files_to_gc
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import (
    _ensure_compatible_indices,
    normalize_arg,
    normalize_args,
    raise_if_indices_overlap,
    validate_partition_keys,
)
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
    write_partition,
)

from ._utils import (
    _cast_categorical_to_index_cat,
    _get_data,
    _maybe_get_categoricals_from_index,
    map_delayed,
)

__all__ = (
    "delete_dataset__delayed",
    "garbage_collect_dataset__delayed",
    "read_dataset_as_delayed",
    "update_dataset_from_delayed",
    "store_delayed_as_dataset",
)


def _delete_all_additional_metadata(dataset_factory):
    delete_indices(dataset_factory=dataset_factory)
    delete_common_metadata(dataset_factory=dataset_factory)


def _delete_tl_metadata(dataset_factory, *args):
    """
    This function serves as a collector function for delayed objects. Therefore
    allowing additional arguments which are not used.
    """
    delete_top_level_metadata(dataset_factory=dataset_factory)


@default_docs
@normalize_args
def delete_dataset__delayed(dataset_uuid=None, store=None, factory=None):
    """
    Parameters
    ----------
    """
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory, load_schema=False,
    )

    gc = garbage_collect_dataset__delayed(factory=dataset_factory)

    mps = dispatch_metapartitions_from_factory(dataset_factory)

    delayed_dataset_uuid = delayed(_delete_all_additional_metadata)(
        dataset_factory=dataset_factory
    )
    mps = map_delayed(
        MetaPartition.delete_from_store,
        mps,
        store=store,
        dataset_uuid=dataset_factory.dataset_uuid,
    )

    return delayed(_delete_tl_metadata)(dataset_factory, mps, gc, delayed_dataset_uuid)


@default_docs
@normalize_args
def garbage_collect_dataset__delayed(
    dataset_uuid: Optional[str] = None,
    store: StoreInput = None,
    chunk_size: int = 100,
    factory=None,
) -> List[Delayed]:
    """
    Remove auxiliary files that are no longer tracked by the dataset.

    These files include indices that are no longer referenced by the metadata
    as well as files in the directories of the tables that are no longer
    referenced. The latter is only applied to static datasets.

    Parameters
    ----------
    chunk_size
        Number of files that should be deleted in a single job.

    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    nested_files = dispatch_files_to_gc(
        dataset_uuid=None, store_factory=None, chunk_size=chunk_size, factory=ds_factory
    )
    return list(
        map_delayed(delete_files, nested_files, store_factory=ds_factory.store_factory)
    )


def _load_and_concat_metapartitions_inner(mps, args, kwargs):
    return MetaPartition.concat_metapartitions(
        [mp.load_dataframes(*args, **kwargs) for mp in mps]
    )


def _load_and_concat_metapartitions(list_of_mps, *args, **kwargs):
    return map_delayed(
        _load_and_concat_metapartitions_inner, list_of_mps, args=args, kwargs=kwargs
    )


@default_docs
@normalize_args
def read_dataset_as_delayed_metapartitions(
    dataset_uuid=None,
    store=None,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals: Optional[Sequence[str]] = None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dispatch_by=None,
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
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    store = ds_factory.store_factory
    mps = dispatch_metapartitions_from_factory(
        dataset_factory=ds_factory, predicates=predicates, dispatch_by=dispatch_by,
    )

    if dispatch_by is not None:
        mps = _load_and_concat_metapartitions(
            mps,
            store=store,
            columns=columns,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )
    else:
        mps = map_delayed(
            MetaPartition.load_dataframes,
            mps,
            store=store,
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

        mps = map_delayed(
            partial(  # type: ignore
                MetaPartition.apply,
                func=partial(  # type: ignore
                    _cast_categorical_to_index_cat, categories=categoricals_from_index
                ),
                type_safe=True,
            ),
            mps,
        )

    return list(mps)


@default_docs
def read_dataset_as_delayed(
    dataset_uuid=None,
    store=None,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals=None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dispatch_by=None,
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
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        dispatch_by=dispatch_by,
    )
    return list(map_delayed(_get_data, mps))


@default_docs
def update_dataset_from_delayed(
    delayed_tasks: List[Delayed],
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
    table_name=SINGLE_TABLE,
):
    """
    A dask.delayed graph to add and store a list of dictionaries containing
    dataframes to a kartothek dataset in store. The input should be a list
    (or splitter pipeline) containing
    :class:`~kartothek.io_components.metapartition.MetaPartition`. If you want to use this
    pipeline step for just deleting partitions without adding new ones you
    have to give an empty meta partition as input (``[Metapartition(None)]``).

    Parameters
    ----------

    See Also
    --------
    :ref:`mutating_datasets`
    """
    partition_on = normalize_arg("partition_on", partition_on)
    store = normalize_arg("store", store)
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
    mps = map_delayed(
        write_partition,
        delayed_tasks,
        secondary_indices=secondary_indices,
        metadata_version=metadata_version,
        partition_on=partition_on,
        store_factory=store,
        df_serializer=df_serializer,
        dataset_uuid=dataset_uuid,
        sort_partitions_by=sort_partitions_by,
        dataset_table_name=table_name,
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
    delayed_tasks: List[Delayed],
    store,
    dataset_uuid=None,
    metadata=None,
    df_serializer=None,
    overwrite=False,
    metadata_merger=None,
    metadata_version=naming.DEFAULT_METADATA_VERSION,
    partition_on=None,
    metadata_storage_format=naming.DEFAULT_METADATA_STORAGE_FORMAT,
    table_name: str = SINGLE_TABLE,
    secondary_indices=None,
) -> Delayed:
    """
    Transform and store a list of dictionaries containing
    dataframes to a kartothek dataset in store.

    Parameters
    ----------
    """
    store = lazy_store(store)
    if dataset_uuid is None:
        dataset_uuid = gen_uuid()

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    raise_if_indices_overlap(partition_on, secondary_indices)

    input_to_mps = partial(
        parse_input_to_metapartition,
        metadata_version=metadata_version,
        table_name=table_name,
    )
    mps = map_delayed(input_to_mps, delayed_tasks)

    if partition_on:
        mps = map_delayed(MetaPartition.partition_on, mps, partition_on=partition_on)

    if secondary_indices:
        mps = map_delayed(MetaPartition.build_indices, mps, columns=secondary_indices)

    mps = map_delayed(
        MetaPartition.store_dataframes,
        mps,
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
