"""
Common code for dask backends.
"""
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterable, Mapping, Optional, Set

import dask.bag as db
from simplekv import KeyValueStore

from kartothek.api.consistency import get_cube_payload
from kartothek.api.discover import discover_datasets, discover_datasets_unchecked
from kartothek.core.cube.constants import (
    KTK_CUBE_DF_SERIALIZER,
    KTK_CUBE_METADATA_STORAGE_FORMAT,
    KTK_CUBE_METADATA_VERSION,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadataBase
from kartothek.core.typing import StoreFactory
from kartothek.io_components.cube.append import check_existing_datasets
from kartothek.io_components.cube.common import check_blocksize, check_store_factory
from kartothek.io_components.cube.query import load_group, plan_query, quick_concat
from kartothek.io_components.cube.remove import (
    prepare_metapartitions_for_removal_action,
)
from kartothek.io_components.cube.write import (
    apply_postwrite_checks,
    check_datasets_prebuild,
    check_datasets_preextend,
    check_provided_metadata_dict,
    multiplex_user_input,
    prepare_data_for_ktk,
    prepare_ktk_metadata,
    prepare_ktk_partition_on,
)
from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import _ensure_compatible_indices
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)
from kartothek.serialization._parquet import ParquetSerializer
from kartothek.utils.ktk_adapters import metadata_factory_from_dataset

__all__ = (
    "append_to_cube_from_bag_internal",
    "build_cube_from_bag_internal",
    "extend_cube_from_bag_internal",
    "query_cube_bag_internal",
)


def ensure_valid_cube_indices(
    existing_datasets: Mapping[str, DatasetMetadataBase], cube: Cube
) -> Cube:
    """
    Parse all existing datasets and infer the required set of indices. We do not
    allow indices to be removed or added in update steps at the momenent and
    need to make sure that existing ones are updated properly.
    The returned `Cube` instance will be a copy of the input with
    `index_columns` and `suppress_index_on` fields adjusted to reflect the
    existing datasets.
    """
    dataset_indices = []
    for ds in existing_datasets.values():
        for internal_table in ds.table_meta:
            dataset_columns = set(ds.table_meta[internal_table].names)
            table_indices = cube.index_columns & dataset_columns
            compatible_indices = _ensure_compatible_indices(ds, table_indices)
            if compatible_indices:
                dataset_indices.append(set(compatible_indices))
    required_indices = cube.index_columns.union(*dataset_indices)
    suppress_index_on = cube.suppress_index_on.difference(*dataset_indices)
    # Need to remove dimension columns since they *are* technically indices but
    # the cube interface class declares them as not indexed just to add them
    # later on, assuming it is not blacklisted
    return cube.copy(
        index_columns=required_indices - set(cube.dimension_columns),
        suppress_index_on=suppress_index_on,
    )


def build_cube_from_bag_internal(
    data: db.Bag,
    cube: Cube,
    store: StoreFactory,
    ktk_cube_dataset_ids: Optional[Iterable[str]],
    metadata: Optional[Dict[str, Dict[str, Any]]],
    overwrite: bool,
    partition_on: Optional[Dict[str, Iterable[str]]],
    df_serializer: Optional[ParquetSerializer] = None,
) -> db.Bag:
    """
    Create dask computation graph that builds a cube with the data supplied from a dask bag.

    Parameters
    ----------
    data: dask.bag.Bag
        Bag containing dataframes
    cube:
        Cube specification.
    store:
        Store to which the data should be written to.
    ktk_cube_dataset_ids:
        Datasets that will be written, must be specified in advance. If left unprovided, it is assumed that only the
        seed dataset will be written.
    metadata:
        Metadata for every dataset.
    overwrite:
        If possibly existing datasets should be overwritten.
    partition_on:
        Optional parition-on attributes for datasets (dictionary mapping :term:`Dataset ID` -> columns).
    df_serializer:
        Optional Dataframe to Parquet serializer

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to build a cube returning the dict of dataset metadata objects.
        The bag has a single partition with a single element.
    """
    check_store_factory(store)

    if ktk_cube_dataset_ids is None:
        ktk_cube_dataset_ids = [cube.seed_dataset]
    else:
        ktk_cube_dataset_ids = sorted(ktk_cube_dataset_ids)

    metadata = check_provided_metadata_dict(metadata, ktk_cube_dataset_ids)
    existing_datasets = discover_datasets_unchecked(cube.uuid_prefix, store)
    check_datasets_prebuild(ktk_cube_dataset_ids, cube, existing_datasets)
    prep_partition_on = prepare_ktk_partition_on(
        cube, ktk_cube_dataset_ids, partition_on
    )
    cube = ensure_valid_cube_indices(existing_datasets, cube)

    data = (
        data.map(multiplex_user_input, cube=cube)
        .map(_check_dataset_ids, ktk_cube_dataset_ids=ktk_cube_dataset_ids)
        .map(
            _multiplex_prepare_data_for_ktk,
            cube=cube,
            existing_payload=set(),
            partition_on=prep_partition_on,
        )
    )

    data = _store_bag_as_dataset_parallel(
        bag=data,
        store=store,
        cube=cube,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata={
            ktk_cube_dataset_id: prepare_ktk_metadata(
                cube, ktk_cube_dataset_id, metadata
            )
            for ktk_cube_dataset_id in ktk_cube_dataset_ids
        },
        overwrite=overwrite,
        update=False,
        existing_datasets=existing_datasets,
        df_serializer=df_serializer,
    )

    data = data.map(
        apply_postwrite_checks,
        cube=cube,
        store=store,
        existing_datasets=existing_datasets,
    )

    return data


def extend_cube_from_bag_internal(
    data: db.Bag,
    cube: Cube,
    store: KeyValueStore,
    ktk_cube_dataset_ids: Optional[Iterable[str]],
    metadata: Optional[Dict[str, Dict[str, Any]]],
    overwrite: bool,
    partition_on: Optional[Dict[str, Iterable[str]]],
    df_serializer: Optional[ParquetSerializer] = None,
) -> db.Bag:
    """
    Create dask computation graph that extends a cube by the data supplied from a dask bag.

    For details on ``data`` and ``metadata``, see :func:`~kartothek.io.eager_cube.build_cube`.

    Parameters
    ----------
    data: dask.bag.Bag
        Bag containing dataframes (see :func:`~kartothek.io.eager_cube.build_cube` for possible format and types).
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store:
        Store to which the data should be written to.
    ktk_cube_dataset_ids:
        Datasets that will be written, must be specified in advance.
    metadata:
        Metadata for every dataset.
    overwrite:
        If possibly existing datasets should be overwritten.
    partition_on:
        Optional parition-on attributes for datasets (dictionary mapping :term:`Dataset ID` -> columns).
    df_serializer:
        Optional Dataframe to Parquet serializer

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to extend a cube returning the dict of dataset metadata objects.
        The bag has a single partition with a single element.
    """
    check_store_factory(store)
    check_datasets_preextend(ktk_cube_dataset_ids, cube)
    if ktk_cube_dataset_ids:
        ktk_cube_dataset_ids = sorted(ktk_cube_dataset_ids)
    else:
        ktk_cube_dataset_ids = []
    metadata = check_provided_metadata_dict(metadata, ktk_cube_dataset_ids)
    prep_partition_on = prepare_ktk_partition_on(
        cube, ktk_cube_dataset_ids, partition_on
    )

    existing_datasets = discover_datasets(cube, store)
    cube = ensure_valid_cube_indices(existing_datasets, cube)
    if overwrite:
        existing_datasets_cut = {
            ktk_cube_dataset_id: ds
            for ktk_cube_dataset_id, ds in existing_datasets.items()
            if ktk_cube_dataset_id not in ktk_cube_dataset_ids
        }
    else:
        existing_datasets_cut = existing_datasets
    existing_payload = get_cube_payload(existing_datasets_cut, cube)

    data = (
        data.map(multiplex_user_input, cube=cube)
        .map(_check_dataset_ids, ktk_cube_dataset_ids=ktk_cube_dataset_ids)
        .map(
            _multiplex_prepare_data_for_ktk,
            cube=cube,
            existing_payload=existing_payload,
            partition_on=prep_partition_on,
        )
    )

    data = _store_bag_as_dataset_parallel(
        bag=data,
        store=store,
        cube=cube,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata={
            ktk_cube_dataset_id: prepare_ktk_metadata(
                cube, ktk_cube_dataset_id, metadata
            )
            for ktk_cube_dataset_id in ktk_cube_dataset_ids
        },
        overwrite=overwrite,
        update=False,
        existing_datasets=existing_datasets,
        df_serializer=df_serializer,
    )

    data = data.map(
        apply_postwrite_checks,
        cube=cube,
        store=store,
        existing_datasets=existing_datasets,
    )

    return data


def query_cube_bag_internal(
    cube,
    store,
    conditions,
    datasets,
    dimension_columns,
    partition_by,
    payload_columns,
    blocksize,
):
    """
    Query cube.

    For detailed documentation, see :func:`~kartothek.io.eager_cube.query_cube`.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: simplekv.KeyValueStore
        KV store that preserves the cube.
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied, optional.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to query, must all be part of the cube. May be either the result of :func:`~kartothek.api.discover.discover_datasets`, a list
        of Ktk_cube dataset ID or ``None`` (in which case auto-discovery will be used).
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
    empty: pandas.DataFrame
        Empty DataFrame with correct dtypes and column order.
    bag: dask.bag.Bag
        Bag of 1-sized partitions of non-empty DataFrames, order by ``partition_by``. Column of DataFrames is
        alphabetically ordered. Data types are provided on best effort (they are restored based on the preserved data,
        but may be different due to Pandas NULL-handling, e.g. integer columns may be floats).
    """
    check_store_factory(store)
    check_blocksize(blocksize)

    intention, empty, groups = plan_query(
        cube=cube,
        store=store,
        conditions=conditions,
        datasets=datasets,
        dimension_columns=dimension_columns,
        partition_by=partition_by,
        payload_columns=payload_columns,
    )

    b = (
        db.from_sequence(seq=groups, partition_size=blocksize)
        .map(load_group, store=store, cube=cube)
        .filter(_not_empty)
    )

    if not intention.partition_by:
        b = (
            b.reduction(
                perpartition=list,
                aggregate=_collect_dfs,
                split_every=False,
                out_type=db.Bag,
            )
            .map(
                _quick_concat_or_none,
                dimension_columns=intention.dimension_columns,
                partition_columns=cube.partition_columns,
            )
            .filter(_not_none)
        )
    return empty, b


def append_to_cube_from_bag_internal(
    data: db.Bag,
    cube: Cube,
    store: StoreFactory,
    ktk_cube_dataset_ids: Optional[Iterable[str]],
    metadata: Optional[Dict[str, Dict[str, Any]]],
    remove_conditions=None,
    df_serializer: Optional[ParquetSerializer] = None,
) -> db.Bag:
    """
    Append data to existing cube.

    For details on ``data`` and ``metadata``, see :func:`~kartothek.io.eager_cube.build_cube`.

    .. important::

        Physical partitions must be updated as a whole. If only single rows within a physical partition are updated, the
        old data is treated as "removed".


    Parameters
    ----------
    data: dask.bag.Bag
        Bag containing dataframes
    cube:
        Cube specification.
    store:
        Store to which the data should be written to.
    ktk_cube_dataset_ids:
        Datasets that will be written, must be specified in advance.
    metadata:
        Metadata for every dataset, optional. For every dataset, only given keys are updated/replaced. Deletion of
        metadata keys is not possible.
    remove_conditions:
        Conditions that select which partitions to remove.
    df_serializer:
        Optional Dataframe to Parquet serializer

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to append to the cube returning the dict of dataset metadata
        objects. The bag has a single partition with a single element.
    """
    check_store_factory(store)
    if ktk_cube_dataset_ids:
        ktk_cube_dataset_ids = sorted(ktk_cube_dataset_ids)
    else:
        ktk_cube_dataset_ids = []
    metadata = check_provided_metadata_dict(metadata, ktk_cube_dataset_ids)

    existing_datasets = discover_datasets(cube, store)
    cube = ensure_valid_cube_indices(existing_datasets, cube)
    # existing_payload is set to empty because we're not checking against any existing payload. ktk will account for the
    # compat check within 1 dataset
    existing_payload: Set[str] = set()

    partition_on = {k: v.partition_keys for k, v in existing_datasets.items()}

    check_existing_datasets(
        existing_datasets=existing_datasets, ktk_cube_dataset_ids=ktk_cube_dataset_ids
    )

    if remove_conditions is not None:
        remove_metapartitions = prepare_metapartitions_for_removal_action(
            cube, store, remove_conditions, ktk_cube_dataset_ids, existing_datasets
        )
        delete_scopes = {
            k: delete_scope for k, (_, _, delete_scope) in remove_metapartitions.items()
        }
    else:
        delete_scopes = {}

    data = (
        data.map(multiplex_user_input, cube=cube)
        .map(_check_dataset_ids, ktk_cube_dataset_ids=ktk_cube_dataset_ids)
        .map(_fill_dataset_ids, ktk_cube_dataset_ids=ktk_cube_dataset_ids)
        .map(
            _multiplex_prepare_data_for_ktk,
            cube=cube,
            existing_payload=existing_payload,
            partition_on=partition_on,
        )
    )

    data = _store_bag_as_dataset_parallel(
        bag=data,
        store=store,
        cube=cube,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
        metadata={
            ktk_cube_dataset_id: prepare_ktk_metadata(
                cube, ktk_cube_dataset_id, metadata
            )
            for ktk_cube_dataset_id in ktk_cube_dataset_ids
        },
        update=True,
        existing_datasets=existing_datasets,
        delete_scopes=delete_scopes,
        df_serializer=df_serializer,
    )

    data = data.map(
        apply_postwrite_checks,
        cube=cube,
        store=store,
        existing_datasets=existing_datasets,
    )

    return data


def _not_empty(df):
    return not df.empty


def _not_none(obj):
    return obj is not None


def _collect_dfs(iter_of_lists):
    dfs = [df for sublist in iter_of_lists for df in sublist]
    return [dfs]


def _quick_concat_or_none(dfs, dimension_columns, partition_columns):
    dfs = list(dfs)
    if dfs:
        return quick_concat(
            dfs=dfs,
            dimension_columns=dimension_columns,
            partition_columns=partition_columns,
        )
    else:
        return None


def _check_dataset_ids(dct, ktk_cube_dataset_ids):
    for ds_name in sorted(dct.keys()):
        if ds_name not in ktk_cube_dataset_ids:
            raise ValueError(
                (
                    'Ktk_cube Dataset ID "{ds_name}" is present during pipeline execution but was not specified in '
                    "ktk_cube_dataset_ids ({ktk_cube_dataset_ids})."
                ).format(
                    ds_name=ds_name,
                    ktk_cube_dataset_ids=", ".join(sorted(ktk_cube_dataset_ids)),
                )
            )

    return dct


def _fill_dataset_ids(dct, ktk_cube_dataset_ids):
    # make sure dct contains an entry for each ktk_cube_dataset_ids, filling in None
    # if necessary
    dct.update({ktk_id: None for ktk_id in ktk_cube_dataset_ids if ktk_id not in dct})
    return dct


def _store_bag_as_dataset_parallel(
    bag: db.Bag,
    store: KeyValueStore,
    cube: Cube,
    ktk_cube_dataset_ids: Iterable[str],
    metadata: Optional[Dict[str, Dict[str, Any]]],
    existing_datasets,
    overwrite: bool = False,
    update: bool = False,
    delete_scopes=None,
    df_serializer: Optional[ParquetSerializer] = None,
) -> db.Bag:
    """
    Vendored, simplified and modified version of kartotheks ``store_bag_as_dataset`` which cannot be easily used to
    store datasets in parallel (e.g. from a dict).

    `delete_scope` is a dictionary mapping the kartothek dataset id to the `delete_scope` of the dataset
    (see `update_dataset_from_partitions` for the definition of the single dataset `delete_scope`).
    """
    if (not update) and (not overwrite):
        for ktk_cube_dataset_id in ktk_cube_dataset_ids:
            raise_if_dataset_exists(
                dataset_uuid=cube.ktk_dataset_uuid(ktk_cube_dataset_id), store=store
            )

    mps = bag.map(_multiplex_parse_input_to_metapartition)

    # prepare_data_for_ktk already runs `MetaPartition.partition_on` and `MetaPartition.build_indices`, so this is not
    # required here anymore

    mps = mps.map(_multiplex_store, store=store, cube=cube, df_serializer=df_serializer)

    aggregate = partial(
        _multiplex_store_dataset_from_partitions_flat,
        cube=cube,
        existing_datasets=existing_datasets,
        metadata=metadata,
        store=store,
        update=update,
        delete_scopes=delete_scopes or {},
    )

    return mps.reduction(
        perpartition=list, aggregate=aggregate, split_every=False, out_type=db.Bag
    )


def _multiplex_prepare_data_for_ktk(data, cube, existing_payload, partition_on):
    result = {}
    for k in sorted(data.keys()):
        v = data.pop(k)
        result[k] = prepare_data_for_ktk(
            v,
            ktk_cube_dataset_id=k,
            cube=cube,
            existing_payload=existing_payload,
            partition_on=partition_on[k],
        )
        del v
    return result


def _multiplex_store_dataset_from_partitions_flat(
    mpss, cube, metadata, update, store, existing_datasets, delete_scopes
):
    dct = defaultdict(list)
    for sublist in mpss:
        for mp in sublist:
            for k, v in mp.items():
                dct[k].append(v)

    result = {}
    for k, v in dct.items():
        if update:
            ds_factory = metadata_factory_from_dataset(
                existing_datasets[k], with_schema=True, store=store
            )
            result[k] = update_dataset_from_partitions(
                v,
                dataset_uuid=cube.ktk_dataset_uuid(k),
                delete_scope=delete_scopes.get(k, []),
                ds_factory=ds_factory,
                metadata=metadata[k],
                metadata_merger=None,
                store_factory=store,
            )
        else:
            result[k] = store_dataset_from_partitions(
                v,
                dataset_metadata=metadata[k],
                dataset_uuid=cube.ktk_dataset_uuid(k),
                metadata_merger=None,
                metadata_storage_format=KTK_CUBE_METADATA_STORAGE_FORMAT,
                store=store,
            )

    # list required for dask.bag
    return [result]


def _multiplex_store(
    data: db.Bag,
    cube: Cube,
    store: StoreFactory,
    df_serializer: Optional[ParquetSerializer] = None,
) -> db.Bag:
    result = {}
    for k in sorted(data.keys()):
        v = data.pop(k)
        result[k] = MetaPartition.store_dataframes(
            v,
            dataset_uuid=cube.ktk_dataset_uuid(k),
            df_serializer=df_serializer or KTK_CUBE_DF_SERIALIZER,
            store=store,
        )
        del v
    return result


def _multiplex_parse_input_to_metapartition(data):
    result = {}
    for k in sorted(data.keys()):
        v = data.pop(k)
        result[k] = parse_input_to_metapartition(
            v, metadata_version=KTK_CUBE_METADATA_VERSION
        )
        del v
    return result
