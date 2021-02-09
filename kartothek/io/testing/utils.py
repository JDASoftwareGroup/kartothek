import math

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile

from kartothek.io.eager import store_dataframes_as_dataset


def create_dataset(dataset_uuid, store_factory, metadata_version):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df.copy(deep=True), df.copy(deep=True)]

    return store_dataframes_as_dataset(
        dfs=df_list,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        secondary_indices="P",
    )


def assert_num_row_groups(store, dataset, part_num_rows, part_chunk_size):
    """
    Assert that the row groups of each partition match the expectation based on the
    number of rows and the chunk size
    """
    # Iterate over the partitions of each index value
    for index, partitions in dataset.indices["p"].index_dct.items():
        for part_key in partitions:
            key = dataset.partitions[part_key].files["table"]
            parquet_file = ParquetFile(store.open(key))
            if part_chunk_size[index] is None:
                assert parquet_file.num_row_groups == 1
            else:
                assert parquet_file.num_row_groups == math.ceil(
                    part_num_rows[index] / part_chunk_size[index]
                )
