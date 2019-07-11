import warnings
from typing import Iterator, List, Union, cast

import pandas as pd

from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.utils import _make_callable


def _index_to_dataframe(idx_name, idx, allowed_labels=None):
    label_col = []
    value_col = []
    for value, labels in idx.items():
        for label in labels:
            if allowed_labels is not None and label not in allowed_labels:
                continue
            label_col.append(label)
            value_col.append(value)
    df = pd.DataFrame({idx_name: value_col, "__partition__": label_col})

    return df


def dispatch_metapartitions_from_factory(
    dataset_factory,
    label_filter=None,
    concat_partitions_on_primary_index=False,
    predicates=None,
    store=None,
    dispatch_by=None,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:
    if not callable(dataset_factory) and not isinstance(
        dataset_factory, DatasetFactory
    ):
        raise TypeError("Need to supply a dataset factory!")

    if dispatch_by and concat_partitions_on_primary_index:
        raise ValueError(
            "Both `dispatch_by` and `concat_partitions_on_primary_index` are provided, "
            "`concat_partitions_on_primary_index` is deprecated and will be removed in the next major release. "
            "Please only provide the `dispatch_by` argument. "
        )
    if concat_partitions_on_primary_index:
        warnings.warn(
            "The keyword `concat_partitions_on_primary_index` is deprecated and will be removed in the next major release. Use `dispatch_by=dataset_factory.partition_keys` to achieve the same behavior instead.",
            DeprecationWarning,
        )
        dispatch_by = dataset_factory.partition_keys

    if dispatch_by and not set(dispatch_by).issubset(
        set(dataset_factory.index_columns)
    ):
        raise RuntimeError(
            f"Dispatch columns must be indexed.\nRequested index: {dispatch_by} but available index columns: {sorted(dataset_factory.index_columns)}"
        )

    if predicates is not None:
        dataset_factory, allowed_labels = _allowed_labels_by_predicates(
            predicates, dataset_factory, dispatch_by
        )
    else:
        allowed_labels = None

    indices_to_dispatch = {
        name: ix.copy(index_dct={})
        for name, ix in dataset_factory.indices.items()
        if isinstance(ix, ExplicitSecondaryIndex)
    }

    if dispatch_by:
        # Build up a DataFrame that contains per row a Partition and its primary index columns.
        base_df = None
        for part_key in dispatch_by:
            dataset_factory.load_index(part_key)
            idx = dataset_factory.indices[part_key].index_dct
            df = _index_to_dataframe(part_key, idx, allowed_labels)
            if base_df is None:
                base_df = df
            else:
                base_df = base_df.merge(df, on=["__partition__"])

        assert base_df is not None
        base_df = cast(pd.DataFrame, base_df)

        # Group the resulting MetaParitions by partition keys or a subset of those keys
        merged_partitions = base_df.groupby(
            by=list(dispatch_by), sort=False, as_index=False
        )
        for group_name, group in merged_partitions:
            if not isinstance(group_name, tuple):
                group_name = (group_name,)
            mps = []
            logical_conjunction = list(
                zip(dispatch_by, ["=="] * len(dispatch_by), group_name)
            )
            for label in group.__partition__:
                mps.append(
                    MetaPartition.from_partition(
                        partition=dataset_factory.partitions[label],
                        dataset_metadata=dataset_factory.metadata,
                        indices=indices_to_dispatch,
                        metadata_version=dataset_factory.metadata_version,
                        table_meta=dataset_factory.table_meta,
                        partition_keys=dataset_factory.partition_keys,
                        logical_conjunction=logical_conjunction,
                    )
                )
            yield mps
    else:

        if allowed_labels is not None:
            partition_labels = allowed_labels
        else:
            partition_labels = dataset_factory.partitions.keys()

        for part_label in partition_labels:

            if label_filter is not None:
                if not label_filter(part_label):
                    continue

            part = dataset_factory.partitions[part_label]

            yield MetaPartition.from_partition(
                partition=part,
                dataset_metadata=dataset_factory.metadata,
                indices=indices_to_dispatch,
                metadata_version=dataset_factory.metadata_version,
                table_meta=dataset_factory.table_meta,
                partition_keys=dataset_factory.partition_keys,
            )


def _allowed_labels_by_predicates(predicates, dataset_factory, dispatch_by):
    if len(predicates) == 0:
        raise ValueError("The behaviour on an empty list of predicates is undefined")

    dataset_factory = dataset_factory.load_partition_indices()

    # Determine the set of columns that are part of a predicate
    columns = set()
    for predicates_inner in predicates:
        if len(predicates_inner) == 0:
            raise ValueError("The behaviour on an empty predicate is undefined")
        for col, _, _ in predicates_inner:
            columns.add(col)

    # Load the necessary indices
    for column in columns:
        if column in dataset_factory.indices:
            dataset_factory = dataset_factory.load_index(column)

    # Narrow down predicates to the columns that have an index.
    # The remaining parts of the predicate are filtered during
    # load_dataframes.
    filtered_predicates = []
    for predicate in predicates:
        new_predicate = []
        for col, op, val in predicate:
            if col in dataset_factory.indices:
                new_predicate.append((col, op, val))
        filtered_predicates.append(new_predicate)

    # In the case that any of the above filters produced an empty predicate,
    # we have to load the full dataset as we cannot prefilter on the indices.
    has_catchall = any(((len(predicate) == 0) for predicate in filtered_predicates))

    # None is a sentinel value for "no predicates"
    allowed_labels = None
    if filtered_predicates and not has_catchall:
        allowed_labels = set()
        for conjunction in filtered_predicates:
            allowed_labels |= _allowed_labels_by_conjunction(
                conjunction, dataset_factory.indices
            )
    return dataset_factory, allowed_labels


def _allowed_labels_by_conjunction(conjunction, indices):
    """
    Returns all partition labels which are allowed by the given conjunction (AND)
    of literals based on the indices

    Parameters
    ----------
    conjunction: list of tuple
        A list of (column, operator, value) tuples
    indices: dict
        A dict column->kartothek.core.index.IndexBase holding the indices to be evaluated
    Returns
    -------
    set: allowed labels
    """
    allowed_by_conjunction = None
    for col, op, val in conjunction:
        allowed_labels = indices[col].eval_operator(op, val)
        if allowed_by_conjunction is not None:
            allowed_by_conjunction &= allowed_labels
        else:
            allowed_by_conjunction = allowed_labels
    return allowed_by_conjunction


def dispatch_metapartitions(
    dataset_uuid,
    store,
    load_dataset_metadata=True,
    keep_indices=True,
    keep_table_meta=True,
    label_filter=None,
    concat_partitions_on_primary_index=False,
    predicates=None,
    dispatch_by=None,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:
    dataset_factory = DatasetFactory(
        dataset_uuid=dataset_uuid,
        store_factory=_make_callable(store),
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=load_dataset_metadata,
    )

    return dispatch_metapartitions_from_factory(
        dataset_factory=dataset_factory,
        label_filter=label_filter,
        predicates=predicates,
        dispatch_by=dispatch_by,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
    )
