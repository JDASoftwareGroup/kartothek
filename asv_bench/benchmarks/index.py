# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import pickle
import shutil
import sys
import tempfile
import uuid
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import storefact

from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.io_components.metapartition import MetaPartition

from .config import AsvBenchmarkConfig


@lru_cache()
def generate_partition_values(num_partitions):
    dataset_uuid = uuid.uuid1().hex
    return [
        f"{dataset_uuid}/column_1={ix % 10}/column_2={ix % 100}/{uuid.uuid1().hex}.parquet"
        for ix in range(num_partitions)
    ]


class IndexBase(AsvBenchmarkConfig):
    def setup(self, number_values, number_partitions, dtype):
        py_type, arrow_type = dtype
        self.partition_values = generate_partition_values(number_partitions)
        index_dct = {
            py_type(val): list(
                np.random.choice(self.partition_values, number_partitions // 2)
            )
            for val in range(number_values)
        }
        self.column_name = "column"
        self.ktk_index = ExplicitSecondaryIndex(
            column=self.column_name, index_dct=index_dct, dtype=arrow_type
        )
        self.tmp_dir = tempfile.mkdtemp()
        self.store = storefact.get_store_from_url("hfs://{}".format(self.tmp_dir))
        self.dataset_uuid = "some_uuid"
        self.storage_key = self.ktk_index.store(self.store, self.dataset_uuid)

        self.ktk_index_not_loaded = ExplicitSecondaryIndex(
            column=self.column_name, index_storage_key=self.storage_key
        )

        self.ktk_index_loaded = self.ktk_index_not_loaded.load(self.store)

    def teardown(self, number_values, number_partitions, dtype):
        shutil.rmtree(self.tmp_dir)


class Index(IndexBase):
    params = (
        [10 * 1, 10 ** 3],  # values
        [10 * 1, 10 ** 3],  # partitions
        [(int, pa.int64())],  # types
    )
    param_names = ["number_values", "number_partitions", "dtype"]

    def time_remove_partitions_inplace(
        self, number_values, number_partitions, arrow_type
    ):
        partitions_to_remove = self.partition_values[len(self.partition_values) // 2 :]
        self.ktk_index.remove_partitions(partitions_to_remove, inplace=True)

    def time_load_index(self, number_values, number_partitions, arrow_type):
        self.ktk_index_not_loaded.load(self.store)

    def time_query_value(self, number_values, number_partitions, arrow_type):
        self.ktk_index.query(number_values / 2)

    def time_as_series(self, number_values, number_partitions, arrow_type):
        self.ktk_index.as_flat_series()

    def time_as_series_partitions_as_index(
        self, number_values, number_partitions, arrow_type
    ):
        self.ktk_index.as_flat_series(partitions_as_index=True)

    def time_observed_values(self, number_values, number_partitions, arrow_type):
        self.ktk_index.observed_values()


class SerializeIndex(IndexBase):
    timeout = 180
    params = (
        [(10 ** 3, 10), (10 ** 4, 100)],  # (values, partitions)
        [(int, pa.int64())],  # types
    )
    param_names = ["number_values__number_partitions", "dtype"]

    def setup(self, number_values__number_partitions, dtype):
        number_values, number_partitions = number_values__number_partitions
        return super().setup(number_values, number_partitions, dtype)

    def teardown(self, number_values__number_partitions, dtype):
        number_values, number_partitions = number_values__number_partitions
        return super().teardown(number_values, number_partitions, dtype)

    def track_mem_serialized(self, number_values__number_partitions, arrow_type):
        # Use `sys.getsizeof` as asv's `mem_*` just reports `0` if memory usage is low
        # enough
        return sys.getsizeof(pickle.dumps(self.ktk_index))

    def time_serialization(self, number_values__number_partitions, arrow_type):
        # Time serialization of indices
        pickle.loads(pickle.dumps(self.ktk_index))


class BuildIndex(AsvBenchmarkConfig):
    params = ([-1, 1], [10 ** 3, 10 ** 4], [10, 100])
    param_names = ["cardinality", "num_values", "partitions_to_merge"]

    def setup(self, cardinality, num_values, partitions_to_merge):
        self.column = "column"
        self.table = "table"
        self.merge_indices = []
        for n in range(partitions_to_merge):
            if cardinality < 0:
                array = ["{:010f}".format(x) for x in np.random.randn(num_values)]
            else:
                unique_vals = ["{:010d}".format(n) for n in range(cardinality)]
                array = [unique_vals[x % len(unique_vals)] for x in range(num_values)]
            self.df = pd.DataFrame({self.column: array})
            self.mp = MetaPartition(
                label=self.table, data={"core": self.df}, metadata_version=4
            )
            self.mp_indices = self.mp.build_indices([self.column])
            self.merge_indices.append(self.mp_indices)

    def time_metapartition_build_index(
        self, cardinality, num_values, partitions_to_merge
    ):
        self.mp.build_indices([self.column])

    def time_merge_indices(self, cardinality, num_values, partitions_to_merge):
        MetaPartition.merge_indices(self.merge_indices)
