import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from typing import Callable, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from simplekv import KeyValueStore

from kartothek.core.factory import DatasetFactory
from kartothek.core.index import ExplicitSecondaryIndex
from kartothek.core.naming import (
    METADATA_BASE_SUFFIX,
    METADATA_FORMAT_JSON,
    TABLE_METADATA_FILE,
)
from kartothek.serialization._io_buffer import BlockBuffer

StoreFactory = Callable[[], KeyValueStore]

_logger = logging.getLogger(__name__)


def get_stats_df(
    dataset_uuid: str, store_factory: StoreFactory, workers: int = 50
) -> pd.DataFrame:
    """
    Get a statistics DataFrame for a given `dataset_uuid` and `store_factory`
    This DataFrame contains statistics the following statistics for each `.parquet` file:
    umber of rows, number of row groups and the file size in bytes.
    It also adds ktk specific statistics like the name of the partition to the DataFrame.

    Parameters
    ----------
    dataset_uuid : str
        uuid of the dataset
    store_factory : StoreFactory
        storefactory of the dataset
    workers : int, optional
        threads or workers used to fetch the data, by default 50

    Returns
    -------
    pd.DataFrame
        A stats DataFrame with the given columns:
        ```
        ["key", "num_rows", "num_row_group", "file_size", "partition]
        ```
    """
    ds_factory = DatasetFactory(dataset_uuid, store_factory)
    ds_keys = list(_get_dataset_keys(ds_factory))
    get_data = partial(_fetch_stats, store_factory)

    with ThreadPoolExecutor(max_workers=workers) as executer:
        data = executer.map(get_data, ds_keys)

    df = pd.DataFrame(data, columns=["key", "num_rows", "num_row_group", "file_size"])
    df = _add_partition_col_to_stats_df(df)

    return df


def _get_dataset_keys(dataset: DatasetFactory) -> Set[str]:
    keys = set()

    # central metadata
    keys.add(dataset.uuid + METADATA_BASE_SUFFIX + METADATA_FORMAT_JSON)

    # common metadata
    for table in dataset.tables:
        keys.add(f"{dataset.uuid}/{table}/{TABLE_METADATA_FILE}")

    # indices
    for index in dataset.indices.values():
        if isinstance(index, ExplicitSecondaryIndex):
            keys.add(str(index.index_storage_key))

    # partition files (usually .parquet files)
    for partition in dataset.partitions.values():
        for f in partition.files.values():
            keys.add(f)

    return keys


def _fetch_stats(
    store_factory: StoreFactory, key: str
) -> Tuple[str, np.float, np.float, np.float]:
    store = store_factory()
    try:
        with closing(BlockBuffer(store.open(key))) as fp:
            _logger.debug(f"Fetching stats for key: {key}")
            if ".parquet" in key:
                fp_parquet = pq.ParquetFile(fp)
                return (
                    key,
                    fp_parquet.metadata.num_rows,
                    fp_parquet.num_row_groups,
                    fp.size,
                )
            else:
                return (key, np.nan, np.nan, fp.size)
    except Exception as e:
        # If an Exception occurs we just report it to the user and return NaN data.
        _logger.warning(f"Could not get stats from {key}: \n {e}")
        return (key, np.nan, np.nan, np.nan)


def _add_partition_col_to_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    # Extract the partion column key and append it as a column to the dataframe.
    return df.assign(partition=df["key"].str.split("/", expand=True)[2])
