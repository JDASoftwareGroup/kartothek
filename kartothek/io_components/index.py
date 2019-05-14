# -*- coding: utf-8 -*-


import logging

from kartothek.core import naming
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import persist_indices

_logger = logging.getLogger(__name__)


def update_indices_from_partitions(partition_list, dataset_metadata_factory):
    """
    This takes indices from a partition list and overwrites all indices in the dataset metadata
    provided by the dataset metadata factory. The same is done in the store dataset part. This is used
    in an additional build index step (by the build_dataset_indices__pipeline) which should be used after
    updating partitions of a dataset.
    """

    dataset_indices = MetaPartition.merge_indices(partition_list)

    indices = persist_indices(
        store=dataset_metadata_factory.store,
        dataset_uuid=dataset_metadata_factory.uuid,
        indices=dataset_indices,
    )

    for column, storage_key in indices.items():
        dataset_metadata_factory.indices[column] = ExplicitSecondaryIndex(
            column=column, index_storage_key=storage_key
        )

    dataset_metadata_factory.store.put(
        naming.metadata_key_from_uuid(dataset_metadata_factory.uuid),
        dataset_metadata_factory.to_json(),
    )
    return dataset_metadata_factory
