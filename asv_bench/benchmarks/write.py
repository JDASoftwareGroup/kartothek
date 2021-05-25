# -*- coding: utf-8 -*-


import tempfile
import uuid

from storefact import get_store_from_url

from kartothek.core.common_metadata import make_meta
from kartothek.core.testing import get_dataframe_alltypes
from kartothek.io_components.metapartition import MetaPartition
from kartothek.io_components.write import (
    persist_common_metadata,
    store_dataset_from_partitions,
)

from .config import AsvBenchmarkConfig


def generate_mp(dataset_metadata=None):
    return MetaPartition(
        label=uuid.uuid4().hex,
        table_meta={"table": make_meta(get_dataframe_alltypes(), origin="alltypes")},
        files={"table": "fakefile"},
        dataset_metadata=dataset_metadata,
    )


def _partition_metadata(depth, num_leafs):
    if depth == 0:
        return "the_end"
    part_meta = _partition_metadata(depth - 1, num_leafs)
    return {str(k): part_meta for k in range(num_leafs)}


def generate_metadata(max_depth=7, num_leafs=5):
    """
    Generate a metadata dictionary which holds many `partition_metadata` keys
    """
    return {
        "creation_time": "2018-05-05 12:00:00",
        "partition_metadata": _partition_metadata(max_depth, num_leafs),
    }


class TimeStoreDataset(AsvBenchmarkConfig):
    timeout = 120
    params = ([10, 10 ** 2, 10 ** 3], [4], [2, 4])
    param_names = ["num_partitions", "max_depth", "num_leafs"]

    def setup(self, num_partitions, max_depth, num_leafs):
        self.store = get_store_from_url("hfs://{}".format(tempfile.mkdtemp()))
        dataset_metadata = generate_metadata(max_depth, num_leafs)
        self.partitions = [generate_mp(dataset_metadata) for _ in range(num_partitions)]
        self.dataset_uuid = "dataset_uuid"
        self.user_dataset_metadata = {}

    def time_store_dataset_from_partitions(self, num_partitions, max_depth, num_leafs):
        store_dataset_from_partitions(
            partition_list=self.partitions,
            store=self.store,
            dataset_uuid=self.dataset_uuid,
            dataset_metadata=self.user_dataset_metadata,
        )


class TimePersistMetadata(AsvBenchmarkConfig):
    timeout = 240
    params = [1, 10 ** 2, 10 ** 3]

    def setup(self, num_partitions):
        self.store = get_store_from_url("hfs://{}".format(tempfile.mkdtemp()))
        self.partitions = [generate_mp() for _ in range(num_partitions)]
        self.dataset_uuid = "dataset_uuid"

    def time_persist_common_metadata(self, num_partitions):
        persist_common_metadata(self.partitions, None, self.store, self.dataset_uuid)
