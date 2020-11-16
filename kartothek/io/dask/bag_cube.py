"""
Dask.Bag IO.
"""
import dask.bag as db

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.io.dask.common_cube import (
    append_to_cube_from_bag_internal,
    build_cube_from_bag_internal,
    extend_cube_from_bag_internal,
    query_cube_bag_internal,
)
from kartothek.io_components.cube.cleanup import get_keys_to_clean
from kartothek.io_components.cube.common import (
    assert_stores_different,
    check_blocksize,
    check_store_factory,
)
from kartothek.io_components.cube.copy import get_copy_keys
from kartothek.io_components.cube.stats import (
    collect_stats_block,
    get_metapartitions_for_stats,
    reduce_stats,
)
from kartothek.utils.ktk_adapters import get_dataset_keys
from kartothek.utils.store import copy_keys

__all__ = (
    "append_to_cube_from_bag",
    "update_cube_from_bag",
    "build_cube_from_bag",
    "cleanup_cube_bag",
    "collect_stats_bag",
    "copy_cube_bag",
    "delete_cube_bag",
    "extend_cube_from_bag",
    "query_cube_bag",
)


def build_cube_from_bag(
    data,
    cube,
    store,
    ktk_cube_dataset_ids=None,
    metadata=None,
    overwrite=False,
    partition_on=None,
):
    """
    Create dask computation graph that builds a cube with the data supplied from a dask bag.

    Parameters
    ----------
    data: dask.Bag
        Bag containing dataframes
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: Callable[[], simplekv.KeyValueStore]
        Store to which the data should be written to.
    ktk_cube_dataset_ids: Optional[Iterable[str]]
        Datasets that will be written, must be specified in advance. If left unprovided, it is assumed that only the
        seed dataset will be written.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset.
    overwrite: bool
        If possibly existing datasets should be overwritten.
    partition_on: Optional[Dict[str, Iterable[str]]]
        Optional parition-on attributes for datasets (dictionary mapping :term:`Dataset ID` -> columns).
        See :ref:`Dimensionality and Partitioning Details` for details.

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to build a cube returning the dict of dataset metadata objects.
        The bag has a single partition with a single element.
    """
    return build_cube_from_bag_internal(
        data=data,
        cube=cube,
        store=store,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata=metadata,
        overwrite=overwrite,
        partition_on=partition_on,
    )


def extend_cube_from_bag(
    data,
    cube,
    store,
    ktk_cube_dataset_ids,
    metadata=None,
    overwrite=False,
    partition_on=None,
):
    """
    Create dask computation graph that extends a cube by the data supplied from a dask bag.

    For details on ``data`` and ``metadata``, see :meth:`build_cube`.

    Parameters
    ----------
    data: dask.Bag
        Bag containing dataframes (see :meth:`build_cube` for possible format and types).
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: simplekv.KeyValueStore
        Store to which the data should be written to.
    ktk_cube_dataset_ids: Optional[Iterable[str]]
        Datasets that will be written, must be specified in advance.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset.
    overwrite: bool
        If possibly existing datasets should be overwritten.
    partition_on: Optional[Dict[str, Iterable[str]]]
        Optional parition-on attributes for datasets (dictionary mapping :term:`Dataset ID` -> columns).
        See :ref:`Dimensionality and Partitioning Details` for details.

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to extend a cube returning the dict of dataset metadata objects.
        The bag has a single partition with a single element.
    """
    return extend_cube_from_bag_internal(
        data=data,
        cube=cube,
        store=store,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata=metadata,
        overwrite=overwrite,
        partition_on=partition_on,
    )


def query_cube_bag(
    cube,
    store,
    conditions=None,
    datasets=None,
    dimension_columns=None,
    partition_by=None,
    payload_columns=None,
    blocksize=1,
):
    """
    Query cube.

    For detailed documentation, see :meth:`query_cube`.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: simplekv.KeyValueStore
        KV store that preserves the cube.
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied, optional.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to query, must all be part of the cube. May be either the result of :meth:`discover_datasets`, an
        iterable of Ktk_cube dataset ID or ``None`` (in which case auto-discovery will be used).
    dimension_columns: Union[None, str, Iterable[str]]
        Dimension columns of the query, may result in projection. If not provided, dimension columns from cube
        specification will be used.
    partition_by: Union[None, str, Iterable[str]]
        By which column logical partitions should be formed. If not provided, a single partition will be generated.
    payload_columns: Union[None, str, Iterable[str]]
        Which columns apart from ``dimension_columns`` and ``partition_by`` should be returned.
    blocksize: int
        Partition size of the bag.

    Returns
    -------
    bag: dask.Bag
        Bag of 1-sized partitions of non-empty DataFrames, order by ``partition_by``. Column of DataFrames is
        alphabetically ordered. Data types are provided on best effort (they are restored based on the preserved data,
        but may be different due to Pandas NULL-handling, e.g. integer columns may be floats).
    """
    _empty, b = query_cube_bag_internal(
        cube=cube,
        store=store,
        conditions=conditions,
        datasets=datasets,
        dimension_columns=dimension_columns,
        partition_by=partition_by,
        payload_columns=payload_columns,
        blocksize=blocksize,
    )
    return b


def delete_cube_bag(cube, store, blocksize=100, datasets=None):
    """
    Delete cube from store.

    .. important::
        This routine only deletes tracked files. Garbage and leftovers from old cubes and failed operations are NOT
        removed.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: Callable[[], simplekv.KeyValueStore]
        KV store.
    blocksize: int
        Number of keys to delete at once.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to delete, must all be part of the cube. May be either the result of :meth:`discover_datasets`, a list
        of Ktk_cube dataset ID or ``None`` (in which case entire cube will be deleted).

    Returns
    -------
    bag: dask.bag.Bag
        A dask bag that performs the given operation. May contain multiple partitions.
    """
    check_store_factory(store)
    check_blocksize(blocksize)

    if not isinstance(datasets, dict):
        datasets = discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=store,
            filter_ktk_cube_dataset_ids=datasets,
        )

    keys = set()
    for ktk_cube_dataset_id in sorted(datasets.keys()):
        ds = datasets[ktk_cube_dataset_id]
        keys |= get_dataset_keys(ds)

    return db.from_sequence(seq=sorted(keys), partition_size=blocksize).map_partitions(
        _delete, store=store
    )


def copy_cube_bag(
    cube, src_store, tgt_store, blocksize=100, overwrite=False, datasets=None
):
    """
    Copy cube from one store to another.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    src_store: Callable[[], simplekv.KeyValueStore]
        Source KV store.
    tgt_store: Callable[[], simplekv.KeyValueStore]
        Target KV store.
    overwrite: bool
        If possibly existing datasets in the target store should be overwritten.
    blocksize: int
        Number of keys to copy at once.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to copy, must all be part of the cube. May be either the result of :meth:`discover_datasets`, a list
        of Ktk_cube dataset ID or ``None`` (in which case entire cube will be copied).

    Returns
    -------
    bag: dask.bag.Bag
        A dask bag that performs the given operation. May contain multiple partitions.
    """
    check_store_factory(src_store)
    check_store_factory(tgt_store)
    check_blocksize(blocksize)
    assert_stores_different(
        src_store, tgt_store, cube.ktk_dataset_uuid(cube.seed_dataset)
    )

    keys = get_copy_keys(
        cube=cube,
        src_store=src_store,
        tgt_store=tgt_store,
        overwrite=overwrite,
        datasets=datasets,
    )

    return db.from_sequence(seq=sorted(keys), partition_size=blocksize).map_partitions(
        copy_keys, src_store=src_store, tgt_store=tgt_store
    )


def collect_stats_bag(cube, store, datasets=None, blocksize=100):
    """
    Collect statistics for given cube.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: simplekv.KeyValueStore
        KV store that preserves the cube.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to query, must all be part of the cube. May be either the result of :meth:`discover_datasets`, a list
        of Ktk_cube dataset ID or ``None`` (in which case auto-discovery will be used).
    blocksize: int
        Number of partitions to scan at once.

    Returns
    -------
    bag: dask.bag.Bag
        A dask bag that returns a single result of the form ``Dict[str, Dict[str, int]]`` and contains statistics per
        ktk_cube dataset ID.
    """
    check_store_factory(store)
    check_blocksize(blocksize)

    if not isinstance(datasets, dict):
        datasets = discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=store,
            filter_ktk_cube_dataset_ids=datasets,
        )

    all_metapartitions = get_metapartitions_for_stats(datasets)

    return (
        db.from_sequence(seq=all_metapartitions, partition_size=blocksize)
        .map_partitions(collect_stats_block, store=store)
        .reduction(
            perpartition=_obj_to_list,
            aggregate=_reduce_stats,
            split_every=False,
            out_type=db.Bag,
        )
    )


def cleanup_cube_bag(cube, store, blocksize=100):
    """
    Remove unused keys from cube datasets.

    .. important::
        All untracked keys which start with the cube's `uuid_prefix` followed by the `KTK_CUBE_UUID_SEPERATOR`
        (e.g. `my_cube_uuid++seed...`) will be deleted by this routine. These keys may be leftovers from past
        overwrites or index updates.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        KV store.
    blocksize: int
        Number of keys to delete at once.

    Returns
    -------
    bag: dask.bag.Bag
        A dask bag that performs the given operation. May contain multiple partitions.
    """
    check_store_factory(store)
    check_blocksize(blocksize)

    store_obj = store()

    datasets = discover_datasets_unchecked(uuid_prefix=cube.uuid_prefix, store=store)
    keys = get_keys_to_clean(cube.uuid_prefix, datasets, store_obj)

    return db.from_sequence(seq=sorted(keys), partition_size=blocksize).map_partitions(
        _delete, store=store
    )


def append_to_cube_from_bag(data, cube, store, ktk_cube_dataset_ids, metadata=None):
    """
    Append data to existing cube.

    For details on ``data`` and ``metadata``, see :meth:`build_cube`.

    .. important::

        Physical partitions must be updated as a whole. If only single rows within a physical partition are updated, the
        old data is treated as "removed".

    .. hint::

        To have better control over the overwrite "mask" (i.e. which partitions are overwritten), you should use
        :meth:`remove_partitions` beforehand or use :meth:`update_cube_from_bag` instead.

    Parameters
    ----------
    data: dask.Bag
        Bag containing dataframes
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: Callable[[], simplekv.KeyValueStore]
        Store to which the data should be written to.
    ktk_cube_dataset_ids: Optional[Iterable[str]]
        Datasets that will be written, must be specified in advance.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset, optional. For every dataset, only given keys are updated/replaced. Deletion of
        metadata keys is not possible.

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to append to the cube returning the dict of dataset metadata
        objects. The bag has a single partition with a single element.
    """
    return append_to_cube_from_bag_internal(
        data=data,
        cube=cube,
        store=store,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata=metadata,
    )


def update_cube_from_bag(
    data, cube, store, remove_conditions, ktk_cube_dataset_ids, metadata=None
):
    """
    Remove partitions and append data to existing cube.

    For details on ``data`` and ``metadata``, see :meth:`build_cube`.

    Only datasets in `ktk_cube_dataset_ids` will be affected.

    Parameters
    ----------
    data: dask.Bag
        Bag containing dataframes
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: Callable[[], simplekv.KeyValueStore]
        Store to which the data should be written to.
    remove_conditions
        Conditions that select the partitions to remove. Must be a condition that only uses
        partition columns.
    ktk_cube_dataset_ids: Optional[Iterable[str]]
        Datasets that will be written, must be specified in advance.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset, optional. For every dataset, only given keys are updated/replaced. Deletion of
        metadata keys is not possible.

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to append to the cube returning the dict of dataset metadata
        objects. The bag has a single partition with a single element.
    """
    return append_to_cube_from_bag_internal(
        data=data,
        cube=cube,
        store=store,
        remove_conditions=remove_conditions,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata=metadata,
    )


def _delete(keys, store):
    if callable(store):
        store = store()

    for k in keys:
        store.delete(k)


def _obj_to_list(obj):
    return [obj]


def _reduce_stats(nested_stats):
    flat = [stats for sub in nested_stats for stats in sub]
    return [reduce_stats(flat)]
