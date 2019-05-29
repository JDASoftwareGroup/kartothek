# -*- coding: utf-8 -*-


from collections import defaultdict

from kartothek.core import naming
from kartothek.core.common_metadata import (
    read_schema_metadata,
    store_schema_metadata,
    validate_compatible,
    validate_shared_columns,
)
from kartothek.core.dataset import DatasetMetadataBuilder
from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
from kartothek.core.partition import Partition
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    partition_labels_from_mps,
)
from kartothek.io_components.utils import (
    _instantiate_store,
    combine_metadata,
    extract_duplicates,
)

SINGLE_CATEGORY = SINGLE_TABLE


def persist_indices(store, dataset_uuid, indices):
    store = _instantiate_store(store)
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
        output_filenames[column] = index.store(store=store, dataset_uuid=dataset_uuid)
    return output_filenames


def persist_common_metadata(partition_list, update_dataset, store, dataset_uuid):
    # hash the schemas for quick equality check with possible false negatives
    # (e.g. other pandas version or null schemas)
    tm_dct = defaultdict(set)
    for mp in partition_list:
        for tab, tm in mp.table_meta.items():
            tm_dct[tab].add(tm)

    if update_dataset:
        if set(tm_dct.keys()) and set(update_dataset.tables) != set(tm_dct.keys()):
            raise ValueError(
                (
                    "Input partitions for update have different tables than dataset:\n"
                    "Input partition tables: {}\n"
                    "Tables of existing dataset: {}"
                ).format(set(tm_dct.keys()), update_dataset.tables)
            )
        for table in update_dataset.tables:
            tm_dct[table].add(
                read_schema_metadata(
                    dataset_uuid=dataset_uuid, store=store, table=table
                )
            )

    result = {}

    # sort tables and schemas to have reproducible error messages
    for table in sorted(tm_dct.keys()):
        schemas = sorted(tm_dct[table], key=lambda s: sorted(s.origin))
        try:
            result[table] = validate_compatible(schemas)
        except ValueError as e:
            raise ValueError(
                "Schemas for table '{table}' of dataset '{dataset_uuid}' are not compatible!\n\n{e}".format(
                    table=table, dataset_uuid=dataset_uuid, e=e
                )
            )

    validate_shared_columns(list(result.values()))

    for table, schema in result.items():
        store_schema_metadata(
            schema=schema, dataset_uuid=dataset_uuid, store=store, table=table
        )
    return result


def store_dataset_from_partitions(
    partition_list,
    store,
    dataset_uuid,
    dataset_metadata=None,
    metadata_merger=None,
    update_dataset=None,
    remove_partitions=None,
    metadata_storage_format=naming.DEFAULT_METADATA_STORAGE_FORMAT,
):
    store = _instantiate_store(store)

    if update_dataset:
        dataset_builder = DatasetMetadataBuilder.from_dataset(update_dataset)
        metadata_version = dataset_builder.metadata_version
    else:
        mp = next(iter(partition_list), None)
        if mp is None:
            raise ValueError(
                "Cannot store empty datasets, partition_list must not be empty if in store mode."
            )

        metadata_version = mp.metadata_version
        dataset_builder = DatasetMetadataBuilder(
            uuid=dataset_uuid,
            metadata_version=metadata_version,
            partition_keys=mp.partition_keys,
        )

    dataset_builder.explicit_partitions = True

    dataset_builder.table_meta = persist_common_metadata(
        partition_list, update_dataset, store, dataset_uuid
    )

    # We can only check for non unique partition labels here and if they occur we will
    # fail hard. The resulting dataset may be corrupted or file may be left in the store
    # without dataset metadata
    partition_labels = partition_labels_from_mps(partition_list)
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
        dataset_builder, metadata_merger, partition_list, dataset_metadata
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


def update_metadata(dataset_builder, metadata_merger, add_partitions, dataset_metadata):

    metadata_list = [dataset_builder.metadata]
    metadata_list += [mp.dataset_metadata for mp in add_partitions]
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
        for sub_mp_dct in mp.metapartitions:
            # label is None in case of an empty partition
            if sub_mp_dct["label"] is not None:
                partition = Partition(
                    label=sub_mp_dct["label"], files=sub_mp_dct["files"]
                )
                dataset_builder.add_partition(sub_mp_dct["label"], partition)

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
        store_instance = _instantiate_store(store)
        for form in ["msgpack", "json"]:
            key = naming.metadata_key_from_uuid(uuid=dataset_uuid, format=form)
            if key in store_instance:
                raise RuntimeError(
                    "Dataset `%s` already exists and overwrite is not permitted!",
                    dataset_uuid,
                )
    except KeyError:
        pass
