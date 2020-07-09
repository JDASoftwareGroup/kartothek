"""
Dask.DataFrame IO.
"""
import dask.bag as db
import dask.dataframe as ddf

from kartothek.io.dask.common_cube import (
    append_to_cube_from_bag_internal,
    build_cube_from_bag_internal,
    extend_cube_from_bag_internal,
    query_cube_bag_internal,
)

__all__ = (
    "append_to_cube_from_dataframe",
    "build_cube_from_dataframe",
    "extend_cube_from_dataframe",
    "query_cube_dataframe",
)


def build_cube_from_dataframe(
    data, cube, store, metadata=None, overwrite=False, partition_on=None
):
    """
    Create dask computation graph that builds a cube with the data supplied from a dask dataframe.

    Parameters
    ----------
    data: Union[dask.DataFrame, Dict[str, dask.DataFrame]
        Data that should be written to the cube. If only a single dataframe is given, it is assumed to be the seed
        dataset.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: Callable[[], simplekv.KeyValueStore]
        Store to which the data should be written to.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset.
    overwrite: bool
        If possibly existing datasets should be overwritten.
    partition_on: Optional[Dict[str, Iterable[str]]]
        Optional parition-on attributes for datasets (dictionary mapping :term:`Dataset ID` -> columns).
        See :ref:`Dimensionality and Partitioning Details` for details.

    Returns
    -------
    metadata_dict: dask.Delayed
        A dask delayed object containing the compute graph to build a cube returning the dict of dataset metadata
        objects.
    """
    data, ktk_cube_dataset_ids = _ddfs_to_bag(data, cube)

    return (
        build_cube_from_bag_internal(
            data=data,
            cube=cube,
            store=store,
            ktk_cube_dataset_ids=ktk_cube_dataset_ids,
            metadata=metadata,
            overwrite=overwrite,
            partition_on=partition_on,
        )
        .map_partitions(_unpack_list, default=None)
        .to_delayed()[0]
    )


def extend_cube_from_dataframe(
    data, cube, store, metadata=None, overwrite=False, partition_on=None
):
    """
    Create dask computation graph that extends a cube by the data supplied from a dask dataframe.

    For details on ``data`` and ``metadata``, see :meth:`build_cube`.

    Parameters
    ----------
    data: Union[dask.DataFrame, Dict[str, dask.DataFrame]
        Data that should be written to the cube. If only a single dataframe is given, it is assumed to be the seed
        dataset.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: simplekv.KeyValueStore
        Store to which the data should be written to.
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
    data, ktk_cube_dataset_ids = _ddfs_to_bag(data, cube)

    return (
        extend_cube_from_bag_internal(
            data=data,
            cube=cube,
            store=store,
            ktk_cube_dataset_ids=ktk_cube_dataset_ids,
            metadata=metadata,
            overwrite=overwrite,
            partition_on=partition_on,
        )
        .map_partitions(_unpack_list, default=None)
        .to_delayed()[0]
    )


def query_cube_dataframe(
    cube,
    store,
    conditions=None,
    datasets=None,
    dimension_columns=None,
    partition_by=None,
    payload_columns=None,
):
    """
    Query cube.

    For detailed documentation, see :meth:`query_cube`.

    .. important::
        In contrast to other backends, the Dask DataFrame may contain partitions with empty DataFrames!

    Parameters
    ----------
    cube: Cube
        Cube specification.
    store: simplekv.KeyValueStore
        KV store that preserves the cube.
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied, optional.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to query, must all be part of the cube. May be either the result of :meth:`discover_datasets`, a list
        of Ktk_cube dataset ID or ``None`` (in which case auto-discovery will be used).
    dimension_columns: Union[None, str, Iterable[str]]
        Dimension columns of the query, may result in projection. If not provided, dimension columns from cube
        specification will be used.
    partition_by: Union[None, str, Iterable[str]]
        By which column logical partitions should be formed. If not provided, a single partition will be generated.
    payload_columns: Union[None, str, Iterable[str]]
        Which columns apart from ``dimension_columns`` and ``partition_by`` should be returned.

    Returns
    -------
    ddf: dask.dataframe.DataFrame
        Dask DataFrame, partitioned and order by ``partition_by``. Column of DataFrames is alphabetically ordered. Data
        types are provided on best effort (they are restored based on the preserved data, but may be different due to
        Pandas NULL-handling, e.g. integer columns may be floats).
    """
    empty, b = query_cube_bag_internal(
        cube=cube,
        store=store,
        conditions=conditions,
        datasets=datasets,
        dimension_columns=dimension_columns,
        partition_by=partition_by,
        payload_columns=payload_columns,
        blocksize=1,
    )

    dfs = b.map_partitions(_unpack_list, default=empty).to_delayed()

    return ddf.from_delayed(
        dfs=dfs, meta=empty, divisions=None  # TODO: figure out an API to support this
    )


def append_to_cube_from_dataframe(data, cube, store, metadata=None):
    """
    Append data to existing cube.

    For details on ``data`` and ``metadata``, see :meth:`build_cube`.

    .. important::

        Physical partitions must be updated as a whole. If only single rows within a physical partition are updated, the
        old data is treated as "removed".

    .. hint::

        To have better control over the overwrite "mask" (i.e. which partitions are overwritten), you should use
        :meth:`remove_partitions` beforehand.

    Parameters
    ----------
    data: dask.Bag
        Bag containing dataframes
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: simplekv.KeyValueStore
        Store to which the data should be written to.
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Metadata for every dataset, optional. For every dataset, only given keys are updated/replaced. Deletion of
        metadata keys is not possible.

    Returns
    -------
    metadata_dict: dask.bag.Bag
        A dask bag object containing the compute graph to append to the cube returning the dict of dataset metadata
        objects. The bag has a single partition with a single element.
    """
    data, ktk_cube_dataset_ids = _ddfs_to_bag(data, cube)

    return (
        append_to_cube_from_bag_internal(
            data=data,
            cube=cube,
            store=store,
            ktk_cube_dataset_ids=ktk_cube_dataset_ids,
            metadata=metadata,
        )
        .map_partitions(_unpack_list, default=None)
        .to_delayed()[0]
    )


def _ddfs_to_bag(data, cube):
    if not isinstance(data, dict):
        data = {cube.seed_dataset: data}

    ktk_cube_dataset_ids = sorted(data.keys())
    bags = []
    for ktk_cube_dataset_id in ktk_cube_dataset_ids:
        bags.append(
            db.from_delayed(data[ktk_cube_dataset_id].to_delayed()).map_partitions(
                _convert_write_bag, ktk_cube_dataset_id=ktk_cube_dataset_id
            )
        )

    return (db.concat(bags), ktk_cube_dataset_ids)


def _unpack_list(l, default):  # noqa
    l = list(l)  # noqa
    if l:
        return l[0]
    else:
        return default


def _convert_write_bag(df, ktk_cube_dataset_id):
    return [{ktk_cube_dataset_id: df}]
