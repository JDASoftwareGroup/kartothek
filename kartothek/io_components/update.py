# -*- coding: utf-8 -*-
"""
This module contains logic to update an existing dataset. Update means adding
new partitions and deleting existing partitions. Kartothek does not allow an
update of the content of existing partitions.
"""


from kartothek.core.index import PartitionIndex
from kartothek.io_components.utils import _instantiate_store
from kartothek.io_components.write import store_dataset_from_partitions


def _get_partitions(dataset, query_params):

    partitions = []
    for params in query_params:
        partitions += dataset.query(**params)

    return partitions


def update_dataset_from_partitions(
    partition_list,
    store_factory,
    dataset_uuid,
    ds_factory,
    delete_scope,
    metadata,
    metadata_merger,
):
    store = _instantiate_store(store_factory)

    if ds_factory:
        ds_factory = ds_factory.load_all_indices()
        remove_partitions = _get_partitions(ds_factory, delete_scope)

        index_columns = list(ds_factory.indices.keys())
        for column in index_columns:
            index = ds_factory.indices[column]
            if isinstance(index, PartitionIndex):
                del ds_factory.indices[column]
    else:
        # Dataset does not exist yet.
        remove_partitions = []

    new_dataset = store_dataset_from_partitions(
        partition_list=partition_list,
        store=store,
        dataset_uuid=dataset_uuid,
        dataset_metadata=metadata,
        metadata_merger=metadata_merger,
        update_dataset=ds_factory,
        remove_partitions=remove_partitions,
    )

    return new_dataset
