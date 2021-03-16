from functools import partial
from typing import Dict, Iterable, List, Optional, cast

from simplekv import KeyValueStore

from kartothek.core import naming
from kartothek.core.common_metadata import (
    SchemaWrapper,
    read_schema_metadata,
    store_schema_metadata,
    validate_compatible,
)
from kartothek.core.dataset import DatasetMetadataBuilder
from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex, IndexBase, PartitionIndex
from kartothek.core.typing import StoreFactory, StoreInput
from kartothek.core.utils import ensure_store
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    MetaPartitionInput,
    parse_input_to_metapartition,
    partition_labels_from_mps,
)
from kartothek.io_components.utils import (
    combine_metadata,
    extract_duplicates,
    sort_values_categorical,
)
from kartothek.serialization import DataFrameSerializer

SINGLE_CATEGORY = SINGLE_TABLE


def write_partition(
    partition_df: MetaPartitionInput,
    secondary_indices: List[str],
    sort_partitions_by: List[str],
    dataset_uuid: str,
    partition_on: List[str],
    store_factory: StoreFactory,
    df_serializer: Optional[DataFrameSerializer],
    metadata_version: int,
    dataset_table_name: str = SINGLE_TABLE,
) -> MetaPartition:
    """
    Write a dataframe to store, performing all necessary preprocessing tasks
    like partitioning, bucketing (NotImplemented), indexing, etc. in the correct order.
    """
    store = ensure_store(store_factory)

    # I don't have access to the group values
    mps = parse_input_to_metapartition(
        partition_df, metadata_version=metadata_version, table_name=dataset_table_name,
    )
    if sort_partitions_by:
        mps = mps.apply(partial(sort_values_categorical, columns=sort_partitions_by))
    if partition_on:
        mps = mps.partition_on(partition_on)
    if secondary_indices:
        mps = mps.build_indices(secondary_indices)
    return mps.store_dataframes(
        store=store, dataset_uuid=dataset_uuid, df_serializer=df_serializer
    )


def persist_indices(
    store: StoreInput, dataset_uuid: str, indices: Dict[str, IndexBase]
) -> Dict[str, str]:
    store = ensure_store(store)
    output_filenames = {}
    for column, index in indices.items():
        # backwards compat
        if isinstance(index, dict):
            legacy_storage_key = "{dataset_uuid}.{column}{suffix}".format(
                dataset_uuid=dataset_uuid,
                column=column,
                suffix=naming.EXTERNAL_INDEX_SUFFIX,
            )
            index = ExplicitSecondaryIndex(
                column=column, index_dct=index, index_storage_key=legacy_storage_key
            )
        elif isinstance(index, PartitionIndex):
            continue
        index = cast(ExplicitSecondaryIndex, index)
        output_filenames[column] = index.store(store=store, dataset_uuid=dataset_uuid)
    return output_filenames


def persist_common_metadata(
    schemas: Iterable[SchemaWrapper],
    update_dataset: Optional[DatasetFactory],
    store: KeyValueStore,
    dataset_uuid: str,
    table_name: str,
):

    if not schemas:
        return None
    schemas_set = set(schemas)
    del schemas

    if update_dataset:
        schemas_set.add(
            read_schema_metadata(
                dataset_uuid=dataset_uuid, store=store, table=table_name
            )
        )

    schemas_sorted = sorted(schemas_set, key=lambda s: sorted(s.origin))

    try:
        result = validate_compatible(schemas_sorted)
    except ValueError as e:
        raise ValueError(
            "Schemas for dataset '{dataset_uuid}' are not compatible!\n\n{e}".format(
                dataset_uuid=dataset_uuid, e=e
            )
        )
    if result:
        store_schema_metadata(
            schema=result, dataset_uuid=dataset_uuid, store=store, table=table_name
        )
    return result


def store_dataset_from_partitions(
    partition_list,
    store: StoreInput,
    dataset_uuid,
    dataset_metadata=None,
    metadata_merger=None,
    update_dataset=None,
    remove_partitions=None,
    metadata_storage_format=naming.DEFAULT_METADATA_STORAGE_FORMAT,
):
    store = ensure_store(store)

    schemas = set()
    if update_dataset:
        dataset_builder = DatasetMetadataBuilder.from_dataset(update_dataset)
        metadata_version = dataset_builder.metadata_version
        table_name = update_dataset.table_name
        schemas.add(update_dataset.schema)
    else:
        mp = next(iter(partition_list), None)

        if mp is None:
            raise ValueError(
                "Cannot store empty datasets, partition_list must not be empty if in store mode."
            )
        table_name = mp.table_name
        metadata_version = mp.metadata_version
        dataset_builder = DatasetMetadataBuilder(
            uuid=dataset_uuid,
            metadata_version=metadata_version,
            partition_keys=mp.partition_keys,
        )

    for mp in partition_list:
        if mp.schema:
            schemas.add(mp.schema)

    dataset_builder.schema = persist_common_metadata(
        schemas=schemas,
        update_dataset=update_dataset,
        store=store,
        dataset_uuid=dataset_uuid,
        table_name=table_name,
    )

    # We can only check for non unique partition labels here and if they occur we will
    # fail hard. The resulting dataset may be corrupted or file may be left in the store
    # without dataset metadata
    partition_labels = partition_labels_from_mps(partition_list)

    # This could be safely removed since we do not allow to set this by the user
    # anymore. It has implications on tests if mocks are used
    non_unique_labels = extract_duplicates(partition_labels)

    if non_unique_labels:
        raise ValueError(
            "The labels {} are duplicated. Dataset metadata was not written.".format(
                ", ".join(non_unique_labels)
            )
        )

    if remove_partitions is None:
        remove_partitions = []

    if metadata_merger is None:
        metadata_merger = combine_metadata

    dataset_builder = update_metadata(
        dataset_builder, metadata_merger, dataset_metadata
    )
    dataset_builder = update_partitions(
        dataset_builder, partition_list, remove_partitions
    )
    dataset_builder = update_indices(
        dataset_builder, store, partition_list, remove_partitions
    )
    if metadata_storage_format.lower() == "json":
        store.put(*dataset_builder.to_json())
    elif metadata_storage_format.lower() == "msgpack":
        store.put(*dataset_builder.to_msgpack())
    else:
        raise ValueError(
            "Unknown metadata storage format encountered: {}".format(
                metadata_storage_format
            )
        )
    dataset = dataset_builder.to_dataset()
    return dataset


def update_metadata(dataset_builder, metadata_merger, dataset_metadata):

    metadata_list = [dataset_builder.metadata]
    new_dataset_metadata = metadata_merger(metadata_list)

    dataset_metadata = dataset_metadata or {}

    if callable(dataset_metadata):
        dataset_metadata = dataset_metadata()

    new_dataset_metadata.update(dataset_metadata)
    for key, value in new_dataset_metadata.items():
        dataset_builder.add_metadata(key, value)
    return dataset_builder


def update_partitions(dataset_builder, add_partitions, remove_partitions):

    for mp in add_partitions:
        for mmp in mp:
            if mmp.label is not None:
                dataset_builder.explicit_partitions = True
                dataset_builder.add_partition(mmp.label, mmp.partition)

    for partition_name in remove_partitions:
        del dataset_builder.partitions[partition_name]

    return dataset_builder


def update_indices(dataset_builder, store, add_partitions, remove_partitions):
    dataset_indices = dataset_builder.indices
    partition_indices = MetaPartition.merge_indices(add_partitions)

    if dataset_indices:  # dataset already exists and will be updated
        if remove_partitions:
            for column, dataset_index in dataset_indices.items():
                dataset_indices[column] = dataset_index.remove_partitions(
                    remove_partitions, inplace=True
                )

        for column, index in partition_indices.items():
            dataset_indices[column] = dataset_indices[column].update(
                index, inplace=True
            )

    else:  # dataset index will be created first time from partitions
        dataset_indices = partition_indices

    # Store indices
    index_filenames = persist_indices(
        store=store, dataset_uuid=dataset_builder.uuid, indices=dataset_indices
    )
    for column, filename in index_filenames.items():
        dataset_builder.add_external_index(column, filename)

    return dataset_builder


def raise_if_dataset_exists(dataset_uuid, store):
    try:
        store_instance = ensure_store(store)
        for form in ["msgpack", "json"]:
            key = naming.metadata_key_from_uuid(uuid=dataset_uuid, format=form)
            if key in store_instance:
                raise RuntimeError(
                    "Dataset `%s` already exists and overwrite is not permitted!",
                    dataset_uuid,
                )
    except KeyError:
        pass
