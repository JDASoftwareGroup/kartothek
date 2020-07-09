"""
Common code to build query functions/pipelines.
"""
import itertools
from functools import reduce

import numpy as np

from kartothek.api.consistency import check_datasets
from kartothek.api.discover import discover_datasets
from kartothek.core.common_metadata import empty_dataframe_from_schema
from kartothek.core.cube.conditions import Conjunction
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io_components.cube.query._group import (
    QueryGroup,
    load_group,
    quick_concat,
)
from kartothek.io_components.cube.query._intention import (
    QueryIntention,
    determine_intention,
)
from kartothek.io_components.cube.query._regroup import regroup
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.utils.ktk_adapters import get_dataset_columns

__all__ = ("QueryGroup", "QueryIntention", "load_group", "plan_query", "quick_concat")


def _get_indexed_columns(datasets):
    """
    Get columns that where indexed by Kartothek.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Available datasets.

    Returns
    -------
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.
    """
    result = {}
    for ktk_cube_dataset_id, ds in datasets.items():
        result[ktk_cube_dataset_id] = set(ds.indices.keys())
    return result


def _load_required_explicit_indices(datasets, intention, store):
    """
    Load indices that are required for query evaluation.

    .. important::
        Primary/partition indices must already be loaded at this point!

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Available datasets.
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.
    store: simplekv.KeyValueStore
        Store to query from.

    Returns
    -------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Available datasets, w/ indices loaded.
    """
    # figure out which columns are required for query planning / regrouping
    requires_columns = reduce(
        set.union,
        (
            cond.columns
            for cond in itertools.chain(
                intention.conditions_pre.values(), intention.conditions_post.values()
            )
        ),
        set(),
    ) | set(intention.partition_by)

    # load all indices that describe these columns
    datasets_result = {}
    for ktk_cube_dataset_id, ds in datasets.items():
        indices = {
            column: index.load(store)
            if (
                isinstance(index, ExplicitSecondaryIndex)
                and (column in requires_columns)
            )
            else index
            for column, index in ds.indices.items()
        }
        ds = ds.copy(indices=indices)
        datasets_result[ktk_cube_dataset_id] = ds

    return datasets_result


def _determine_restrictive_dataset_ids(cube, datasets, intention):
    """
    Determine which datasets are restrictive.

    These are datasets which contain non-dimension columns and non-partition columns to which users wishes to apply
    restrictions (via conditions or via partition-by).

    Parameters
    ----------
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Available datasets.
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.

    Returns
    -------
    restrictive_dataset_ids: Set[str]
        Set of restrictive datasets (by Ktk_cube dataset ID).
    """
    result = set()
    for ktk_cube_dataset_id, dataset in datasets.items():
        if ktk_cube_dataset_id == cube.seed_dataset:
            continue

        mask = (
            set(intention.partition_by)
            | intention.conditions_pre.get(ktk_cube_dataset_id, Conjunction([])).columns
            | intention.conditions_post.get(
                ktk_cube_dataset_id, Conjunction([])
            ).columns
        ) - (set(cube.dimension_columns) | set(cube.partition_columns))
        overlap = mask & get_dataset_columns(dataset)
        if overlap:
            result.add(ktk_cube_dataset_id)

    return result


def _dermine_load_columns(cube, datasets, intention):
    """
    Determine which columns to load from given datasets.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Available datasets.
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.

    Returns
    -------
    load_columns: Dict[str, Set[str]]
        Columns to load.
    """
    result = {}
    for ktk_cube_dataset_id, ds in datasets.items():
        is_seed = ktk_cube_dataset_id == cube.seed_dataset
        ds_cols = get_dataset_columns(ds)
        dimensionality = ds_cols & set(cube.dimension_columns)
        is_projection = not dimensionality.issubset(set(intention.dimension_columns))

        mask = (
            set(intention.output_columns)
            | set(intention.dimension_columns)
            | intention.conditions_post.get(
                ktk_cube_dataset_id, Conjunction([])
            ).columns
        )
        if not is_seed:
            # optimize load routine by only restore partition columns for seed
            mask -= set(cube.partition_columns)

        candidates = ds_cols & mask
        payload = candidates - set(cube.partition_columns) - set(cube.dimension_columns)
        payload_requested = len(payload) > 0

        if is_seed or payload_requested:
            if is_projection and payload_requested:
                raise ValueError(
                    (
                        'Cannot project dataset "{ktk_cube_dataset_id}" with dimensionality [{dimensionality}] to '
                        "[{dimension_columns}] while keeping the following payload intact: {payload}"
                    ).format(
                        ktk_cube_dataset_id=ktk_cube_dataset_id,
                        dimensionality=", ".join(sorted(dimensionality)),
                        dimension_columns=", ".join(
                            sorted(intention.dimension_columns)
                        ),
                        payload=", ".join(sorted(payload)),
                    )
                )

            result[ktk_cube_dataset_id] = candidates
    return result


def _filter_relevant_datasets(datasets, load_columns):
    """
    Filter datasets so only ones that actually load columns are left.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets to filter.
    load_columns: Dict[str, Set[str]]
        Columns to load.

    Returns
    -------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Filtered datasets.
    """
    which = set(load_columns.keys())
    return {
        ktk_cube_dataset_id: ds
        for ktk_cube_dataset_id, ds in datasets.items()
        if ktk_cube_dataset_id in which
    }


def _reduce_empty_dtype_sizes(df):
    """
    Try to find smaller dtypes for empty DF.

    Currently, the following conversions are implemented:

    - all integers to ``int8``
    - all floats to ``float32``

    Parameters
    ----------
    df: pandas.DataFrame
        Empty DataFrame, will be modified.

    Returns
    -------
    df: pandas.DataFrame
        Empty DataFrame w/ smaller types.
    """

    def _reduce_dtype(dtype):
        if np.issubdtype(dtype, np.signedinteger):
            return np.int8
        elif np.issubdtype(dtype, np.unsignedinteger):
            return np.uint8
        elif np.issubdtype(dtype, np.floating):
            return np.float32
        else:
            return dtype

    return df.astype({col: _reduce_dtype(df[col].dtype) for col in df.columns})


def plan_query(
    conditions, cube, datasets, dimension_columns, partition_by, payload_columns, store,
):
    """
    Plan cube query execution.

    .. important::
        If the intention does not contain a partition-by, this partition by the cube partition columns to speed up the
        query on parallel backends. In that case, the backend must concat and check the resulting dataframes before
        passing it to the user.

    Parameters
    ----------
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied.
    cube: Cube
        Cube specification.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to query, must all be part of the cube.
    dimension_columns: Optional[Iterable[str]]
        Dimension columns of the query, may result in projection.
    partition_by: Optional[Iterable[str]]
        By which column logical partitions should be formed.
    payload_columns: Optional[Iterable[str]]
        Which columns apart from ``dimension_columns`` and ``partition_by`` should be returned.
    store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Store to query from.

    Returns
    -------
    intent: QueryIntention
        Query intention.
    empty_df: pandas.DataFrame
        Empty DataFrame representing the output types.
    groups: Tuple[QueryGroup]
        Tuple of query groups. May be empty.
    """
    if callable(store):
        store = store()

    if not isinstance(datasets, dict):
        datasets = discover_datasets(
            cube=cube, store=store, filter_ktk_cube_dataset_ids=datasets
        )
    else:
        datasets = check_datasets(datasets, cube)

    datasets = {
        ktk_cube_dataset_id: ds.load_partition_indices()
        for ktk_cube_dataset_id, ds in datasets.items()
    }
    indexed_columns = _get_indexed_columns(datasets)

    intention = determine_intention(
        cube=cube,
        datasets=datasets,
        dimension_columns=dimension_columns,
        partition_by=partition_by,
        conditions=conditions,
        payload_columns=payload_columns,
        indexed_columns=indexed_columns,
    )

    datasets = _load_required_explicit_indices(datasets, intention, store)

    restrictive_dataset_ids = _determine_restrictive_dataset_ids(
        cube=cube, datasets=datasets, intention=intention
    )

    load_columns = _dermine_load_columns(
        cube=cube, datasets=datasets, intention=intention
    )

    datasets = _filter_relevant_datasets(datasets=datasets, load_columns=load_columns)

    empty_df = {
        ktk_cube_dataset_id: _reduce_empty_dtype_sizes(
            empty_dataframe_from_schema(
                schema=ds.table_meta[SINGLE_TABLE],
                columns=sorted(
                    get_dataset_columns(ds) & set(load_columns[ktk_cube_dataset_id])
                ),
            )
        )
        for ktk_cube_dataset_id, ds in datasets.items()
    }

    empty_df_single = empty_df[cube.seed_dataset].copy()
    for k, df in empty_df.items():
        if k == cube.seed_dataset:
            continue
        if empty_df_single is None:
            empty_df_single = df.copy()
        else:
            empty_df_single = empty_df_single.merge(df)
    empty_df_single = empty_df_single[list(intention.output_columns)]

    groups = regroup(
        intention,
        cube=cube,
        datasets=datasets,
        empty_df=empty_df,
        indexed_columns=indexed_columns,
        load_columns=load_columns,
        restrictive_dataset_ids=restrictive_dataset_ids,
    )
    return intention, empty_df_single, groups
