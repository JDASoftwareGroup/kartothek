"""
Tooling to quickly discover datasets in a given blob store.
"""
import logging
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from kartothek.api.consistency import check_datasets
from kartothek.core.cube.constants import (
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
    KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    KTK_CUBE_UUID_SEPERATOR,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.naming import (
    METADATA_BASE_SUFFIX,
    METADATA_FORMAT_JSON,
    METADATA_FORMAT_MSGPACK,
)
from kartothek.core.typing import StoreInput
from kartothek.core.utils import ensure_store
from kartothek.utils.converters import converter_str_set_optional

__all__ = (
    "discover_cube",
    "discover_datasets",
    "discover_datasets_unchecked",
    "discover_ktk_cube_dataset_ids",
)


_logger = logging.getLogger(__name__)


def _discover_dataset_meta_files(prefix: str, store: StoreInput) -> Set[str]:
    """
    Get meta file names for all datasets.

    Parameters
    ----------
    prefix
        the prefix.
    store
        KV store.

    Returns
    -------
    names: Set[str]
        The meta file names
    """

    store = ensure_store(store)

    names = {
        name[: -len(METADATA_BASE_SUFFIX + suffix)]
        for name in store.iter_prefixes(delimiter="/", prefix=prefix)
        for suffix in [METADATA_FORMAT_JSON, METADATA_FORMAT_MSGPACK]
        if name.endswith(METADATA_BASE_SUFFIX + suffix)
    }
    return names


def discover_ktk_cube_dataset_ids(uuid_prefix: str, store: StoreInput) -> Set[str]:
    """
    Get ktk_cube dataset ids for all datasets.

    Parameters
    ----------
    uuid_prefix
        Dataset UUID prefix.
    store
        KV store.

    Returns
    -------
    names: Set[str]
        The ktk_cube dataset ids

    """
    prefix = uuid_prefix + KTK_CUBE_UUID_SEPERATOR
    names = _discover_dataset_meta_files(prefix, store)
    return set([name[len(prefix) :] for name in names])


def discover_datasets_unchecked(
    uuid_prefix: str,
    store: StoreInput,
    filter_ktk_cube_dataset_ids: Optional[Union[str, Iterable[str]]] = None,
) -> Dict[str, DatasetMetadata]:
    """
    Get all known datasets that may belong to a give cube w/o applying any checks.

    .. warning::
        The results are not checked for validity. Found datasets may be incompatible w/ the given cube. Use
        :meth:`check_datasets` to check the results, or go for :meth:`discover_datasets` in the first place.

    Parameters
    ----------
    uuid_prefix
        Dataset UUID prefix.
    store
        KV store.
    filter_ktk_cube_dataset_ids
        Optional selection of datasets to include.

    Returns
    -------
    datasets: Dict[str, DatasetMetadata]
        All discovered datasets. Empty Dict if no dataset is found
    """
    store = ensure_store(store)

    filter_ktk_cube_dataset_ids = converter_str_set_optional(
        filter_ktk_cube_dataset_ids
    )
    prefix = uuid_prefix + KTK_CUBE_UUID_SEPERATOR

    names = _discover_dataset_meta_files(prefix, store)

    if filter_ktk_cube_dataset_ids is not None:
        names = {
            name for name in names if name[len(prefix) :] in filter_ktk_cube_dataset_ids
        }

    result = {}
    # sorted iteration for determistic error messages in case DatasetMetadata.load_from_store fails
    for name in sorted(names):
        try:
            result[name[len(prefix) :]] = DatasetMetadata.load_from_store(
                uuid=name, store=store, load_schema=True, load_all_indices=False
            )
        except KeyError as e:
            _logger.warning(
                'Ignore dataset "{name}" due to KeyError: {e}'.format(name=name, e=e)
            )

    return result


def discover_datasets(
    cube: Cube,
    store: StoreInput,
    filter_ktk_cube_dataset_ids: Optional[Union[str, Iterable[str]]] = None,
) -> Dict[str, DatasetMetadata]:
    """
    Get all known datasets that belong to a give cube.

    Parameters
    ----------
    cube
        Cube specification.
    store
        KV store.
    filter_ktk_cube_dataset_ids
        Optional selection of datasets to include.

    Returns
    -------
    datasets: Dict[str, DatasetMetadata]
        All discovered datasets.

    Raises
    ------
    ValueError
        In case no valid cube could be discovered.
    """
    filter_ktk_cube_dataset_ids = converter_str_set_optional(
        filter_ktk_cube_dataset_ids
    )
    result = discover_datasets_unchecked(
        cube.uuid_prefix, store, filter_ktk_cube_dataset_ids
    )
    if filter_ktk_cube_dataset_ids is not None:
        if isinstance(filter_ktk_cube_dataset_ids, str):
            filter_ktk_cube_dataset_ids = {filter_ktk_cube_dataset_ids}
        else:
            filter_ktk_cube_dataset_ids = set(filter_ktk_cube_dataset_ids)
        missing = filter_ktk_cube_dataset_ids - set(result.keys())
        if missing:
            raise ValueError(
                "Could not find the following requested datasets: {missing}".format(
                    missing=", ".join(sorted(missing))
                )
            )
    check_datasets(result, cube)

    return result


def discover_cube(
    uuid_prefix: str,
    store: StoreInput,
    filter_ktk_cube_dataset_ids: Optional[Union[str, Iterable[str]]] = None,
) -> Tuple[Cube, Dict[str, DatasetMetadata]]:
    """
    Recover cube information from store.

    Parameters
    ----------
    uuid_prefix
        Dataset UUID prefix.
    store
        KV store.
    filter_ktk_cube_dataset_ids
        Optional selection of datasets to include.

    Returns
    -------
    cube: Cube
        Cube specification.
    datasets: Dict[str, DatasetMetadata]
        All discovered datasets.
    """
    datasets = discover_datasets_unchecked(
        uuid_prefix, store, filter_ktk_cube_dataset_ids
    )

    seed_candidates = {
        ktk_cube_dataset_id
        for ktk_cube_dataset_id, ds in datasets.items()
        if ds.metadata.get(
            KTK_CUBE_METADATA_KEY_IS_SEED, ds.metadata.get("klee_is_seed", False)
        )
    }
    if len(seed_candidates) == 0:
        raise ValueError(
            'Could not find seed dataset for cube "{uuid_prefix}".'.format(
                uuid_prefix=uuid_prefix
            )
        )
    elif len(seed_candidates) > 1:
        raise ValueError(
            'Found multiple possible seed datasets for cube "{uuid_prefix}": {seed_candidates}'.format(
                uuid_prefix=uuid_prefix,
                seed_candidates=", ".join(sorted(seed_candidates)),
            )
        )
    seed_dataset = list(seed_candidates)[0]

    seed_ds = datasets[seed_dataset]
    dimension_columns = seed_ds.metadata.get(
        KTK_CUBE_METADATA_DIMENSION_COLUMNS,
        seed_ds.metadata.get("klee_dimension_columns"),
    )
    if dimension_columns is None:
        raise ValueError(
            'Could not recover dimension columns from seed dataset ("{seed_dataset}") of cube "{uuid_prefix}".'.format(
                seed_dataset=seed_dataset, uuid_prefix=uuid_prefix
            )
        )

    # datasets written with new kartothek versions (after merge of PR#7747)
    # always set KTK_CUBE_METADATA_PARTITION_COLUMNS and "klee_timestamp_column" in the metadata.
    # Older versions of ktk_cube do not write these; instead, these columns are inferred from
    # the actual partitioning: partition_columns are all but the last partition key
    #
    # TODO: once we're sure we have re-written all kartothek cubes, the code
    # in the branch `if partition_columns is None` below can be removed.
    #
    # read the now unused timestamp column just to make sure we can still read older cubes.
    #
    # TODO: once all cubes are re-created and don't use timestamp column anymore, remove the timestamp column handling
    #       entirely
    partition_columns = seed_ds.metadata.get(
        KTK_CUBE_METADATA_PARTITION_COLUMNS,
        seed_ds.metadata.get("klee_partition_columns"),
    )
    timestamp_column = seed_ds.metadata.get("klee_timestamp_column")

    if partition_columns is None:
        # infer the partition columns and timestamp column from the actual partitioning:
        partition_keys = seed_ds.partition_keys
        if len(partition_keys) == 0:
            raise ValueError(
                'Seed dataset ("{seed_dataset}") has no partition keys.'.format(  # type: ignore # noqa
                    seed_dataset=seed_dataset, partition_keys=", ".join(partition_keys),
                )
            )
        elif len(partition_keys) < 2:
            raise ValueError(
                (
                    'Seed dataset ("{seed_dataset}") has only a single partition key ({partition_key}) '
                    "but should have at least 2."
                ).format(seed_dataset=seed_dataset, partition_key=partition_keys[0])
            )
        partition_columns = partition_keys[:-1]
        timestamp_column = partition_keys[-1]

    index_columns = set()
    for ds in datasets.values():
        index_columns |= set(ds.indices.keys()) - (
            set(dimension_columns) | set(partition_columns) | {timestamp_column}
        )

    # we only support the default timestamp column in the compat code
    if (timestamp_column is not None) and (timestamp_column != "KLEE_TS"):
        raise NotImplementedError(
            f"Can only read old cubes if the timestamp column is 'KLEE_TS', but '{timestamp_column}' was detected."
        )

    cube = Cube(
        uuid_prefix=uuid_prefix,
        dimension_columns=dimension_columns,
        partition_columns=partition_columns,
        index_columns=index_columns,
        seed_dataset=seed_dataset,
        suppress_index_on=seed_ds.metadata.get(KTK_CUBE_METADATA_SUPPRESS_INDEX_ON),
    )

    datasets = check_datasets(datasets, cube)
    return cube, datasets
