from typing import Iterator, List, Optional, Set, Union, cast, overload

import pandas as pd

from kartothek.core.factory import DatasetFactory
from kartothek.core.typing import StoreInput
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.utils import normalize_args
from kartothek.serialization import (
    PredicatesType,
    check_predicates,
    columns_in_predicates,
)


@overload
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory,
    predicates: PredicatesType = None,
    dispatch_by: None = None,
) -> Iterator[MetaPartition]:
    ...


@overload
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory, predicates: PredicatesType, dispatch_by: List[str],
) -> Iterator[List[MetaPartition]]:
    ...


@normalize_args
def dispatch_metapartitions_from_factory(
    dataset_factory: DatasetFactory,
    predicates: PredicatesType = None,
    dispatch_by: Optional[List[str]] = None,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:
    """

    :meta private:
    """

    if dispatch_by is not None and not set(dispatch_by).issubset(
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

    if dispatch_by is not None:
        base_df = cast(pd.DataFrame, base_df)

        if len(dispatch_by) == 0:
            merged_partitions = [((""), base_df)]
        else:
            # Group the resulting MetaParitions by partition keys or a subset of those keys
            merged_partitions = base_df.groupby(
                by=list(dispatch_by), sort=True, as_index=False
            )

        for group_name, group in merged_partitions:
            if not isinstance(group_name, tuple):
                group_name = (group_name,)  # type: ignore
            mps = []
            logical_conjunction = list(
                zip(dispatch_by, ["=="] * len(dispatch_by), group_name)
            )
            for label in group.index.unique():
                mps.append(
                    MetaPartition.from_partition(
                        partition=dataset_factory.partitions[label],
                        metadata_version=dataset_factory.metadata_version,
                        schema=dataset_factory.schema,
                        partition_keys=dataset_factory.partition_keys,
                        logical_conjunction=logical_conjunction,
                        table_name=dataset_factory.table_name,
                    )
                )
            yield mps
    else:
        for part_label in base_df.index.unique():
            part = dataset_factory.partitions[part_label]

            yield MetaPartition.from_partition(
                partition=part,
                metadata_version=dataset_factory.metadata_version,
                schema=dataset_factory.schema,
                partition_keys=dataset_factory.partition_keys,
                table_name=dataset_factory.table_name,
            )


def dispatch_metapartitions(
    dataset_uuid: str,
    store: StoreInput,
    predicates: PredicatesType = None,
    dispatch_by: Optional[List[str]] = None,
) -> Union[Iterator[MetaPartition], Iterator[List[MetaPartition]]]:
    dataset_factory = DatasetFactory(
        dataset_uuid=dataset_uuid,
        store_factory=store,
        load_schema=True,
        load_all_indices=False,
    )

    return dispatch_metapartitions_from_factory(
        dataset_factory=dataset_factory, predicates=predicates, dispatch_by=dispatch_by,
    )
