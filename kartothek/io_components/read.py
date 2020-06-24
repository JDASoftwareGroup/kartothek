import warnings
from typing import Callable, Iterator, List, Optional, Set, Union, cast, overload

import pandas as pd
from simplekv import KeyValueStore

from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.utils import _make_callable, normalize_args
from kartothek.serialization import (
    PredicatesType,
    check_predicates,
    columns_in_predicates,
)


@overload
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory,
    label_filter: Optional[Callable] = None,
    concat_partitions_on_primary_index: bool = False,
    predicates: PredicatesType = None,
    store: Optional[Callable[[], KeyValueStore]] = None,
    dispatch_by: None = None,
    dispatch_metadata: bool = False,
) -> Iterator[MetaPartition]:
    ...


@overload
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory,
    label_filter: Optional[Callable],
    concat_partitions_on_primary_index: bool,
    predicates: PredicatesType,
    store: Optional[Callable[[], KeyValueStore]],
    dispatch_by: List[str],
    dispatch_metadata: bool,
) -> Iterator[List[MetaPartition]]:
    ...


@normalize_args
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory,
    label_filter: Optional[Callable] = None,
    concat_partitions_on_primary_index: bool = False,
    predicates: PredicatesType = None,
    store: Optional[Callable[[], KeyValueStore]] = None,
    dispatch_by: Optional[List[str]] = None,
    dispatch_metadata: bool = False,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:

    if dispatch_metadata:

        warnings.warn(
            "The dispatch of metadata and index information as part of the MetaPartition instance is deprecated. "
            "The future behaviour will be that this metadata is not dispatched. To set the future behaviour, "
            "specifiy ``dispatch_metadata=False``",
            DeprecationWarning,
        )

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
    check_predicates(predicates)

    # Determine which indices need to be loaded.
    index_cols: Set[str] = set()
    if dispatch_by:
        index_cols |= set(dispatch_by)

    if predicates:
        predicate_cols = set(columns_in_predicates(predicates))
        predicate_index_cols = predicate_cols & set(dataset_factory.index_columns)
        index_cols |= predicate_index_cols

    for col in index_cols:
        dataset_factory.load_index(col)

    base_df = dataset_factory.get_indices_as_dataframe(
        list(index_cols), predicates=predicates
    )

    if label_filter:
        base_df = base_df[base_df.index.map(label_filter)]

    indices_to_dispatch = {
        name: ix.unload()
        for name, ix in dataset_factory.indices.items()
        if isinstance(ix, ExplicitSecondaryIndex)
    }

    if dispatch_by:
        base_df = cast(pd.DataFrame, base_df)

        # Group the resulting MetaParitions by partition keys or a subset of those keys
        merged_partitions = base_df.groupby(
            by=list(dispatch_by), sort=True, as_index=False
        )
        for group_name, group in merged_partitions:
            if not isinstance(group_name, tuple):
                group_name = (group_name,)
            mps = []
            logical_conjunction = list(
                zip(dispatch_by, ["=="] * len(dispatch_by), group_name)
            )
            for label in group.index.unique():
                mps.append(
                    MetaPartition.from_partition(
                        partition=dataset_factory.partitions[label],
                        dataset_metadata=dataset_factory.metadata
                        if dispatch_metadata
                        else None,
                        indices=indices_to_dispatch if dispatch_metadata else None,
                        metadata_version=dataset_factory.metadata_version,
                        table_meta=dataset_factory.table_meta,
                        partition_keys=dataset_factory.partition_keys,
                        logical_conjunction=logical_conjunction,
                    )
                )
            yield mps
    else:
        for part_label in base_df.index.unique():
            part = dataset_factory.partitions[part_label]

            yield MetaPartition.from_partition(
                partition=part,
                dataset_metadata=dataset_factory.metadata
                if dispatch_metadata
                else None,
                indices=indices_to_dispatch if dispatch_metadata else None,
                metadata_version=dataset_factory.metadata_version,
                table_meta=dataset_factory.table_meta,
                partition_keys=dataset_factory.partition_keys,
            )


def dispatch_metapartitions(
    dataset_uuid: str,
    store: Union[KeyValueStore, Callable[[], KeyValueStore]],
    load_dataset_metadata: bool = True,
    keep_indices: bool = True,
    keep_table_meta: bool = True,
    label_filter: Optional[Callable] = None,
    concat_partitions_on_primary_index: bool = False,
    predicates: PredicatesType = None,
    dispatch_by: Optional[List[str]] = None,
    dispatch_metadata: bool = False,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:
    dataset_factory = DatasetFactory(
        dataset_uuid=dataset_uuid,
        store_factory=cast(Callable[[], KeyValueStore], _make_callable(store)),
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=load_dataset_metadata,
    )

    return dispatch_metapartitions_from_factory(
        dataset_factory=dataset_factory,
        store=None,
        label_filter=label_filter,
        predicates=predicates,
        dispatch_by=dispatch_by,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        dispatch_metadata=dispatch_metadata,
    )
