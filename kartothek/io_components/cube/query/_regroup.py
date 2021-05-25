"""
Code that implements dataset alignment and partition-by.
"""
from collections import defaultdict
from copy import copy

import pandas as pd

from kartothek.core.cube.conditions import C, Conjunction
from kartothek.io_components.cube.query._group import QueryGroup
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.utils.ktk_adapters import (
    get_partition_dataframe,
    metadata_factory_from_dataset,
)
from kartothek.utils.pandas import aggregate_to_lists, merge_dataframes_robust

__all__ = ("regroup",)


def _labels_col(ktk_cube_dataset_id):
    """
    Column that is used internally to track labels present for a given dataset.

    Parameters
    ----------
    ktk_cube_dataset_id: str
        Ktk_cube Dataset ID:

    Returns
    -------
    labels_col: str
        Column name.
    """
    return "__ktk_cube_labels_{}".format(ktk_cube_dataset_id)


def _aligned_df_to_label2gp(df, datasets, group_id, label2gp):
    """
    Transfer data from aligned DataFrame to "label to group+partition"-map per dataset.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame w/ labels column per datasets as per :meth:`_labels_col`.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are processed by the regrouper.
    group_id: int
        ID of the current group.
    label2gp: Dict[str, Dict[str, Tuple[int, int]]]
        Maps "dataset ID -> (label -> (group ID, partition ID))", will be modified.
    """
    uuids = sorted(datasets.keys())
    cols = [_labels_col(ktk_cube_dataset_id) for ktk_cube_dataset_id in uuids]
    for partition_id, labeldata in enumerate(zip(*[df[col].values for col in cols])):
        for labels, ktk_cube_dataset_id in zip(labeldata, uuids):
            if not isinstance(labels, list):
                assert pd.isnull(labels)
                continue
            for label in labels:
                label2gp[ktk_cube_dataset_id][label].append((group_id, partition_id))


def _create_dataset_df(
    preconditions, ktk_cube_dataset_id, ds, cube, local_partition_by
):
    """
    Create DataFrame per dataset w/ partition information.

    The output will have a single row per partition that shares the same physical partition and the same partition-by
    attributes. For this, the following columns are present:

    - ``'__ktk_cube_labels_<ktk_cube dataset ID>'``: for this dataset, contains lists of labels for the partition entry
      partition entry
    - physical partition columns
    - additional partition-by columns (if available)

    Parameters
    ----------
    preconditions: Conjunction
        Pre-conditions to be applied to this dataset.
    ktk_cube_dataset_id: str
        Dataset ID.
    ds: kartothek.core.dataset.DatasetMetadata
        Dataset.
    cube: Cube
        Cube specification.
    local_partition_by: Tuple[str, ...]
        Partition-by columns, if available.

    Returns
    -------
    dataset_df: pandas.DataFrame
        Dataset DF.
    """
    preconditions = preconditions.split_by_column()
    all_ktk_cube_partition_cols = sorted(
        set(local_partition_by) | (set(ds.partition_keys) & set(cube.partition_columns))
    )

    # build DF based on partition data
    df = get_partition_dataframe(dataset=ds, cube=cube)
    df.index.rename(_labels_col(ktk_cube_dataset_id), inplace=True)

    # apply pre-conditions
    entries = set(df.index.values)
    for pcol, conj in preconditions.items():
        index_series = ds.indices[pcol].as_flat_series(
            partitions_as_index=True, compact=False
        )
        index_df = pd.DataFrame({pcol: index_series}, index=index_series.index)
        index_df = conj.filter_df(index_df)
        entries &= set(index_df.index.values)
    df = df.loc[sorted(entries)].reset_index(drop=False)

    # add partition_by data (except ds.partition_keys since they are already present)
    for pcol in sorted(set(local_partition_by) - set(ds.partition_keys)):
        series = ds.indices[pcol].as_flat_series(
            partitions_as_index=True, compact=False
        )
        partition_df = pd.DataFrame(
            {pcol: series.values, _labels_col(ktk_cube_dataset_id): series.index.values}
        )
        df = df.merge(partition_df, on=_labels_col(ktk_cube_dataset_id))

    # non-partition indices are not required anymore
    df = df.loc[
        :, [_labels_col(ktk_cube_dataset_id)] + all_ktk_cube_partition_cols
    ].copy()

    # compactify labels
    df = aggregate_to_lists(
        df, all_ktk_cube_partition_cols, _labels_col(ktk_cube_dataset_id)
    )

    return df


def _create_aligned_partition_df(
    datasets, cube, intention, indexed_columns, restrictive_dataset_ids
):
    """
    Create DataFrame w/ aligned partitions.

    The output will have a single row per partition that shares the same physical partition and the same partition-by
    attributes. For this, the following columns are present:

    - ``'__ktk_cube_labels_<ktk_cube dataset ID>'``: a column per dataset w/ either NULL or a list of labels that belong to the
      partition entry
    - physical partition columns
    - additional partition-by columns

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are processed by the regrouper.
    cube: Cube
        Cube specification.
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.
    restrictive_dataset_ids: Set[str]
        Datasets (by Ktk_cube dataset ID) that are restrictive during the join process.

    Returns
    -------
    df_aligned: pandas.DataFrame
        Aligned partitions-DF.
    """
    # Stage 1: Partition DataFrames per Dataset.
    #
    # These DataFrames are classified in 3 categories:
    # - seed: seed dataset
    # - restrict: conditions are applied (therefore data must be present)
    # - other: not a seed and w/o any condition
    df_seed = None
    dfs_restrict = []
    dfs_other = []

    for ktk_cube_dataset_id, ds in datasets.items():
        preconditions = intention.conditions_pre.get(
            ktk_cube_dataset_id, Conjunction([])
        )
        local_partition_by = sorted(
            indexed_columns[ktk_cube_dataset_id] & set(intention.partition_by)
        )
        df = _create_dataset_df(
            preconditions=preconditions,
            ktk_cube_dataset_id=ktk_cube_dataset_id,
            ds=ds,
            cube=cube,
            local_partition_by=local_partition_by,
        )

        # categorize
        if ktk_cube_dataset_id == cube.seed_dataset:
            assert df_seed is None
            df_seed = df
        elif ktk_cube_dataset_id in restrictive_dataset_ids:
            dfs_restrict.append(df)
        else:
            dfs_other.append(df)

    # Stage 2: Alignment
    #
    # Partition DataFrames are aligned based on Cube.partition_columns and their category.
    assert df_seed is not None
    df_aligned = df_seed
    for df_join in dfs_restrict:
        df_aligned = merge_dataframes_robust(df_aligned, df_join, how="inner")
    for df_join in dfs_other:
        df_aligned = merge_dataframes_robust(df_aligned, df_join, how="left")

    return df_aligned


def _regroup(df_aligned, intention, indexed_columns, datasets, cube):
    """
    Based on partition_by, form query groups.

    .. important::
        If tine intention does not contain a partition-by, this partition by the cube partition columns to speed up the
        query on parallel backends. In that case, the backend must concat and check the resulting dataframes before
        passing it to the user.

    Parameters
    ----------
    df_aligned: pandas.DataFrame
        aligned DataFrame, taken from :meth:`_create_aligned_partition_df`
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are processed by the regrouper.
    cube: Cube
        Cube specification.

    Returns
    -------
    label2gp: Dict[str, Dict[str, Tuple[int, int]]]
        Maps "dataset ID -> (label -> (group ID, partition ID))".
    group2cond: Dict[int, kartothek.core.cube.conditions.Conjunction]
        Condition per group.
    """
    partition_by = intention.partition_by
    if not partition_by:
        # special code to speed up the query
        partition_by = cube.partition_columns

    label2gp = defaultdict(lambda: defaultdict(list))
    group2cond = {}
    # figure out which datasets are affected by which additional condition
    extra_conditions_target = {}
    for ktk_cube_dataset_id, cols in indexed_columns.items():
        if ktk_cube_dataset_id not in datasets:
            # may be irrelevant
            continue
        for col in cols & set(partition_by):
            extra_conditions_target[col] = ktk_cube_dataset_id

    # generate groups
    for g, df_g in df_aligned.groupby(list(partition_by), sort=True):
        gid = g
        if len(partition_by) == 1:
            g = (g,)

        conditions_g = copy(intention.conditions_post)
        for g_part, col in zip(g, partition_by):
            if col in cube.partition_columns:
                # we do not need predicate pushdown for physical partition columns
                continue

            ktk_cube_dataset_id = extra_conditions_target[col]
            conditions_g[ktk_cube_dataset_id] = conditions_g.get(
                ktk_cube_dataset_id, Conjunction([])
            ) & (C(col) == g_part)

        _aligned_df_to_label2gp(df_g, datasets, gid, label2gp)
        group2cond[gid] = conditions_g

    return label2gp, group2cond


def _map_ktk_mps_to_groups(cube, datasets, label2gp):
    """
    Map Kartothek metapartitions to groups.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are processed by the regrouper.
    label2gp: Dict[str, Dict[str, Tuple[int, int]]]
        Maps "dataset ID -> (label -> (group ID, partition ID))".

    Returns
    -------
    groups: Dict[int, Dict[int, Dict[str, Tuple[kartothek.io_components.metapartition.MetaPartition, ...]]]]
        Maps "group ID -> (partition ID -> (dataset ID -> list of MetaPartitions))"
    """
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ktk_cube_dataset_id, ds in datasets.items():
        label2gp_sub = label2gp[ktk_cube_dataset_id]
        for mp in dispatch_metapartitions_from_factory(
            dataset_factory=metadata_factory_from_dataset(ds),
            concat_partitions_on_primary_index=False,
        ):
            if mp.label not in label2gp_sub:
                # filtered out by pre-condition
                continue
            for group_id, partition_id in label2gp_sub[mp.label]:
                groups[group_id][partition_id][ktk_cube_dataset_id].append(mp)

    return groups


def regroup(
    intention,
    cube,
    datasets,
    empty_df,
    indexed_columns,
    load_columns,
    restrictive_dataset_ids,
):
    """
    Align and regroup partitions.

    Parameters
    ----------
    intention: kartothek.io_components.cube.query._intention.QueryIntention
        Query intention.
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are relevant (i.e. have columns that should be loaded).
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.
    empty_df: Dict[str, pandas.DataFrame]
        Empty DataFrame for each dataset ID.
    load_columns: Dict[str, Set[str]]
        Columns to load.
    restrictive_dataset_ids: Set[str]
        Datasets (by Ktk_cube dataset ID) that are restrictive during the join process.

    Returns
    -------
    groups: Tuple[QueryGroup, ...]
        Query groups in correct order.
    """
    df_aligned = _create_aligned_partition_df(
        datasets=datasets,
        cube=cube,
        intention=intention,
        indexed_columns=indexed_columns,
        restrictive_dataset_ids=restrictive_dataset_ids,
    )
    label2gp, group2cond = _regroup(
        df_aligned=df_aligned,
        intention=intention,
        indexed_columns=indexed_columns,
        datasets=datasets,
        cube=cube,
    )

    groups = _map_ktk_mps_to_groups(cube=cube, datasets=datasets, label2gp=label2gp)

    result = []
    for group_id in sorted(groups.keys()):
        # strip defaultdicts here
        metapartitions = {
            partition_id: {
                ktk_cube_dataset_id: mps for ktk_cube_dataset_id, mps in sub.items()
            }
            for partition_id, sub in groups[group_id].items()
        }
        predicates = {
            ktk_cube_dataset_id: [cond.predicate]
            for ktk_cube_dataset_id, cond in group2cond[group_id].items()
        }
        result.append(
            QueryGroup(
                metapartitions=metapartitions,
                load_columns=load_columns,
                output_columns=intention.output_columns,
                predicates=predicates,
                empty_df=empty_df,
                dimension_columns=intention.dimension_columns,
                restrictive_dataset_ids=restrictive_dataset_ids,
            )
        )

    return tuple(result)
