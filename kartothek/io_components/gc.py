# -*- coding: utf-8 -*-


from kartothek.core.factory import _ensure_factory
from kartothek.core.naming import TABLE_METADATA_FILE


def dispatch_files_to_gc(dataset_uuid, store_factory, chunk_size, factory):
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store_factory,
        factory=factory,
        load_dataset_metadata=False,
    )
    dataset_uuid = dataset_uuid or ds_factory.uuid

    index_path = "{dataset_uuid}/indices/".format(dataset_uuid=dataset_uuid)
    remove_index_files = set(ds_factory.store.iter_keys(prefix=index_path))

    for index in ds_factory.indices.values():
        index_keys = set()
        # We only add the indices that are saved as explicit indices
        if index.index_storage_key:
            index_keys.add(index.index_storage_key)
        remove_index_files -= index_keys

    remove_table_files = set()
    if ds_factory.explicit_partitions:
        table_files = set()
        for partition in ds_factory.partitions.values():
            for name in partition.files.values():
                table_files.add(name)

        for table in ds_factory.tables:
            table_path = "{dataset_uuid}/{table}/".format(
                dataset_uuid=dataset_uuid, table=table
            )
            table_files.add(table_path + TABLE_METADATA_FILE)
            for key in ds_factory.store.iter_keys(prefix=table_path):
                remove_table_files.add(key)
        remove_table_files -= table_files

    files_to_remove = list(remove_index_files | remove_table_files)

    if chunk_size is None:
        yield files_to_remove
    else:
        for i in range(0, len(files_to_remove), chunk_size):
            yield files_to_remove[i : i + chunk_size]


def delete_files(files, store_factory):
    store = store_factory()
    for f in files:
        store.delete(f)
