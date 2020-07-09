"""
Query group code datastructure and load code.
"""
import typing

import attr
import pandas as pd

from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition
from kartothek.utils.converters import converter_str
from kartothek.utils.pandas import (
    concat_dataframes,
    drop_sorted_duplicates_keep_last,
    sort_dataframe,
)

__all__ = ("QueryGroup", "load_group", "quick_concat")


@attr.s(frozen=True)
class QueryGroup:
    """
    Query group, aka logical partition w/ all kartothek metapartition and information required to load the data.

    Parameters
    ----------
    metapartition: Dict[int, Dict[str, Tuple[kartothek.io_components.metapartition.MetaPartition, ...]]]
        Mapping from partition ID to metapartitions per dataset ID.
    load_columns: Dict[str, Set[str]]
        Columns to load.
    output_columns: Tuple[str, ...]
        Tuple of columns that will be returned from the query API.
    predicates: Dict[str, Tuple[Tuple[Tuple[str, str, Any], ...], ...]]
        Predicates for each dataset ID.
    empty_df: Dict[str, pandas.DataFrame]
        Empty DataFrame for each dataset ID.
    dimension_columns: Tuple[str, ...]
        Dimension columns, used for de-duplication and to join data.
    restrictive_dataset_ids: Set[str]
        Datasets (by Ktk_cube dataset ID) that are restrictive during the join process.
    """

    metapartitions = attr.ib(
        type=typing.Dict[int, typing.Dict[str, typing.Tuple[MetaPartition, ...]]]
    )
    load_columns = attr.ib(type=typing.Dict[str, typing.Tuple[str, ...]])
    output_columns = attr.ib(type=typing.Tuple[str, ...])
    predicates = attr.ib(
        type=typing.Dict[
            str,
            typing.Tuple[typing.Tuple[typing.Tuple[str, str, typing.Any], ...], ...],
        ]
    )
    empty_df = attr.ib(type=typing.Dict[str, pd.DataFrame])
    dimension_columns = attr.ib(type=typing.Tuple[str, ...])
    restrictive_dataset_ids = attr.ib(type=typing.Set[str])


def _load_all_mps(mps, store, load_columns, predicates, empty):
    """
    Load kartothek_cube-relevant data from all given MetaPartitions.

    The result will be a concatenated Dataframe.

    Parameters
    ----------
    mps: Iterable[MetaPartition]
        MetaPartitions to load.
    store: simplekv.KeyValueStore
        Store to load data from.
    load_columns: List[str]
        Columns to load.
    predicates: Optional[List[List[Tuple[str, str, Any]]]]
        Predicates to apply during load.
    empty: pandas.DataFrame
        Empty Dataframe dummy.

    Returns
    -------
    df: pandas.DataFrame
        Concatenated data.
    """
    dfs_mp = []
    for mp in mps:
        mp = mp.load_dataframes(
            store=store,
            predicate_pushdown_to_io=True,
            tables=[SINGLE_TABLE],
            columns={SINGLE_TABLE: sorted(load_columns)},
            predicates=predicates,
        )
        df = mp.data[SINGLE_TABLE]
        df.columns = df.columns.map(converter_str)
        dfs_mp.append(df)
    return concat_dataframes(dfs_mp, empty)


def _load_partition_dfs(cube, group, partition_mps, store):
    """
    Load partition Dataframes for seed, restrictive and other data.

    The information about the merge strategy (seed, restricting, others) is taken from ``group``.

    Parameters
    ----------
    cube: Cube
        Cube spec.
    group: QueryGroup
        Query group.
    partition_mps: Dict[str, Iterable[MetaPartition]]
        MetaPartitions for every dataset in this partition.
    store: simplekv.KeyValueStore
        Store to load data from.

    Returns
    -------
    df_seed: pandas.DataFrame
        Seed data.
    dfs_restrict: List[pandas.DataFrame]
        Restrictive data (for inner join).
    dfs_other: List[pandas.DataFrame]
        Other data (for left join).
    """
    df_seed = None
    dfs_restrict = []
    dfs_other = []

    for ktk_cube_dataset_id, empty in group.empty_df.items():
        mps = partition_mps.get(ktk_cube_dataset_id, [])
        df = _load_all_mps(
            mps=mps,
            store=store,
            load_columns=list(group.load_columns[ktk_cube_dataset_id]),
            predicates=group.predicates.get(ktk_cube_dataset_id, None),
            empty=empty,
        )

        # de-duplicate and sort data
        # PERF: keep order of dimensionality identical to group.dimension_columns
        df_cols = set(df.columns)
        dimensionality = [c for c in group.dimension_columns if c in df_cols]
        df = sort_dataframe(df=df, columns=dimensionality)

        df = drop_sorted_duplicates_keep_last(df, dimensionality)

        if ktk_cube_dataset_id == cube.seed_dataset:
            assert df_seed is None
            df_seed = df
        elif ktk_cube_dataset_id in group.restrictive_dataset_ids:
            dfs_restrict.append(df)
        else:
            dfs_other.append(df)

    assert df_seed is not None
    return df_seed, dfs_restrict, dfs_other


def _load_partition(cube, group, partition_mps, store):
    """
    Load partition and merge partition data within given QueryGroup.

    The information about the merge strategy (seed, restricting, others) is taken from ``group``.

    Parameters
    ----------
    cube: Cube
        Cube spec.
    group: QueryGroup
        Query group.
    partition_mps: Dict[str, Iterable[MetaPartition]]
        MetaPartitions for every dataset in this partition.
    store: simplekv.KeyValueStore
        Store to load data from.

    Returns
    -------
    df: pandas.DataFrame
        Merged data.
    """
    # MEMORY: keep the DF references only as long as they are required:
    #         - use only 1 "intermediate result variable" called df_partition
    #         - consume the DFs lists (dfs_restrict, dfs_other) while iterating over them
    df_partition, dfs_restrict, dfs_other = _load_partition_dfs(
        cube=cube, group=group, partition_mps=partition_mps, store=store
    )

    while dfs_restrict:
        df_partition = df_partition.merge(dfs_restrict.pop(0), how="inner")
    while dfs_other:
        df_partition = df_partition.merge(dfs_other.pop(0), how="left")

    return df_partition.loc[:, list(group.output_columns)]


def load_group(group, store, cube):
    """
    Load :py:class:`QueryGroup` and return DataFrame.

    Parameters
    ----------
    group: QueryGroup
        Query group.
    store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        Store to load data from.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe, may be empty.
    """
    if callable(store):
        store = store()

    partition_results = []
    for partition_id in sorted(group.metapartitions.keys()):
        partition_results.append(
            _load_partition(
                cube=cube,
                group=group,
                partition_mps=group.metapartitions[partition_id],
                store=store,
            )
        )

    # concat all partitions
    return quick_concat(
        dfs=partition_results,
        dimension_columns=group.dimension_columns,
        partition_columns=cube.partition_columns,
    )


def quick_concat(dfs, dimension_columns, partition_columns):
    """
    Fast version of::

        pd.concat(
            dfs,
            ignore_index=True,
            sort=False,
        ).sort_values(dimension_columns + partition_columns).reset_index(drop=True)

    if inputs are presorted.

    Parameters
    -----------
    dfs: Iterable[pandas.DataFrame]
        DataFrames to concat.
    dimension_columns: Iterable[str]
        Dimension columns in correct order.
    partition_columns: Iterable[str]
        Partition columns in correct order.

    Returns
    -------
    df: pandas.DataFrame
        Concatenated result.
    """
    return sort_dataframe(
        df=concat_dataframes(dfs),
        columns=list(dimension_columns) + list(partition_columns),
    )
