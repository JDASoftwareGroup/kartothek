"""
Methods to make working with Kartothek easier.
"""
from __future__ import absolute_import

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
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.serialization._io_buffer import BlockBuffer
from kartothek.utils.converters import converter_str

__all__ = (
    "get_dataset_columns",
    "get_dataset_keys",
    "get_dataset_schema",
    "get_partition_dataframe",
    "get_physical_partition_stats",
    "metadata_factory_from_dataset",
)


def get_dataset_schema(dataset):
    """
    Get schema from a Kartothek_Cube-compatible Kartothek dataset.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        Dataset to get the schema from.

    Returns
    -------
    schema: pyarrow.Schema
        Schema data.
    """
    return dataset.table_meta[SINGLE_TABLE]


def get_dataset_columns(dataset):
    """
    Get columns present in a Kartothek_Cube-compatible Kartothek dataset.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        Dataset to get the columns from.

    Returns
    -------
    columns: Set[str]
        Usable columns.
    """
    return {
        converter_str(col)
        for col in get_dataset_schema(dataset).names
        if not col.startswith("__") and col != "KLEE_TS"
    }


def get_dataset_keys(dataset):
    """
    Get store keys that belong to the given Kartothek dataset.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        Datasets to scan for keys.

    Returns
    -------
    keys: Set[str]
        Storage keys.
    """
    keys = set()

    # central metadata
    keys.add(dataset.uuid + METADATA_BASE_SUFFIX + METADATA_FORMAT_JSON)

    # common metadata
    for table in dataset.tables:
        keys.add("{}/{}/{}".format(dataset.uuid, table, TABLE_METADATA_FILE))

    # indices
    for index in dataset.indices.values():
        if isinstance(index, ExplicitSecondaryIndex):
            keys.add(index.index_storage_key)

    # partition files (usually .parquet files)
    for partition in dataset.partitions.values():
        for f in partition.files.values():
            keys.add(f)

    return keys


class _DummyStore(KeyValueStore):
    """
    Dummy store that should not be used.
    """

    pass


def _dummy_store_factory():
    """
    Creates unusable dummy store.
    """
    return _DummyStore()


def metadata_factory_from_dataset(dataset, with_schema=True, store=None):
    """
    Create :py:class:`DatasetFactory` from :py:class:`DatasetMetadata`.

    Parameters
    ----------
    dataset: DatasetMetadata
        Already loaded dataset.
    with_schema: bool
        If dataset was loaded with ``load_schema``.
    store: Optional[Callable[[], simplekv.KeyValueStore]]
        Optional store factory.

    Returns
    -------
    factory: DatasetFactory
        Metadata factory w/ caches pre-filled.
    """
    factory = DatasetFactory(
        dataset_uuid=dataset.uuid,
        store_factory=store or _dummy_store_factory,
        load_schema=with_schema,
    )
    factory._cache_metadata = dataset
    factory.is_loaded = True
    return factory


def get_physical_partition_stats(metapartitions, store):
    """
    Get statistics for partition.

    .. hint::
        To get the metapartitions pre-aligned, use ``concat_partitions_on_primary_index=True`` during dispatch.

    Parameters
    ----------
    metapartitions: Iterable[kartothek.io_components.metapartition.MetaPartition]
        Iterable of metapartitions belonging to the same physical partition.
    store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        KV store.

    Returns
    -------
    stats: Dict[str, int]
        Statistics for the current partition.
    """
    if callable(store):
        store = store()

    files = 0
    blobsize = 0
    rows = 0
    for mp in metapartitions:
        for f in mp.files.values():
            files += 1
            fp = BlockBuffer(store.open(f))
            try:
                fp_parquet = pq.ParquetFile(fp)
                rows += fp_parquet.metadata.num_rows
                blobsize += fp.size
            finally:
                fp.close()

    return {"blobsize": blobsize, "files": files, "partitions": 1, "rows": rows}


def get_partition_dataframe(dataset, cube):
    """
    Create DataFrame that represent the partioning of the dataset.

    The row index named ``"partition"`` include the partition labels, the columns are the physical partition columns.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        Dataset to analyze, with partition indices pre-loaded.
    cube: kartothek.core.cube.cube.Cube
        Cube spec.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with partition data.
    """
    cols = sorted(set(dataset.partition_keys) - {"KLEE_TS"})

    if not cols:
        return pd.DataFrame(
            index=pd.Index(sorted(dataset.partitions.keys()), name="partition")
        )

    series_list = []
    for pcol in cols:
        series_list.append(
            dataset.indices[pcol].as_flat_series(
                partitions_as_index=True, compact=False
            )
        )
    return (
        pd.concat(series_list, axis=1, sort=False)
        .sort_index()
        .rename_axis(index="partition")
    )
