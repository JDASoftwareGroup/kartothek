# -*- coding: utf-8 -*-


from kartothek.core import naming
from kartothek.core.naming import metadata_key_from_uuid


def delete_common_metadata(dataset_factory):
    for table in dataset_factory.tables:
        key = "{}/{}/{}".format(dataset_factory.uuid, table, naming.TABLE_METADATA_FILE)
        dataset_factory.store.delete(key)
    return dataset_factory


def delete_indices(dataset_factory):
    for index_object in dataset_factory.indices.values():
        index_key = index_object.index_storage_key
        dataset_factory.store.delete(index_key)
    return dataset_factory


def delete_top_level_metadata(dataset_factory, *args):
    """
    The additional arguments allow to schedule this function with delayed objects.
    """
    dataset_factory.store.delete(metadata_key_from_uuid(dataset_factory.dataset_uuid))
