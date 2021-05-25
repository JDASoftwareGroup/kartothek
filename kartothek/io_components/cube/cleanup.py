from functools import reduce

from kartothek.core.cube.constants import KTK_CUBE_UUID_SEPERATOR
from kartothek.utils.ktk_adapters import get_dataset_keys

__all__ = ("get_keys_to_clean",)


def get_keys_to_clean(cube_uuid_prefix, datasets, store):
    """
    Get the keys that are present in the store but can be deleted.

    Parameters
    ----------
    store: simplekv.KeyValueStore
        KV store.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets to scan for keys.

    Returns
    -------
    keys: Set[str]
        Keys to delete.
    """
    keys_should = reduce(
        set.union, (get_dataset_keys(ds) for ds in datasets.values()), set()
    )

    keys_present = {
        k for k in store.iter_keys(cube_uuid_prefix + KTK_CUBE_UUID_SEPERATOR)
    }

    return keys_present - keys_should
