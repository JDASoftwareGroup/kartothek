from __future__ import absolute_import

from copy import copy
from typing import Callable, Dict, Iterable, Optional, Union

from minimalkv import KeyValueStore

from kartothek.api.discover import check_datasets, discover_datasets_unchecked
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.utils.ktk_adapters import get_dataset_keys

__all__ = (
    "get_copy_keys",
    "get_datasets_to_copy",
)


def get_datasets_to_copy(
    cube: Cube,
    src_store: Union[Callable[[], KeyValueStore], KeyValueStore],
    tgt_store: Union[Callable[[], KeyValueStore], KeyValueStore],
    overwrite: bool,
    datasets: Optional[Union[Iterable[str], Dict[str, DatasetMetadata]]] = None,
) -> Dict[str, DatasetMetadata]:
    """
    Determine all dataset names of a given cube that should be copied and apply addtional consistency checks.
    Copying only a specific set of datasets is possible by providing a list of dataset names via the parameter `datasets`.

    Parameters
    ----------
    cube:
        Cube specification.
    src_store:
        Source KV store.
    tgt_store:
        Target KV store.
    overwrite:
        If possibly existing datasets in the target store should be overwritten.
    datasets:
        Datasets to copy, must all be part of the cube. May be either the result of :func:`~kartothek.api.discover.discover_datasets`, an
        iterable of Ktk_cube dataset ID or ``None`` (in which case entire cube will be copied).

    Returns
    -------
    all_datasets: Dict[str, DatasetMetadata]
        All datasets that should be copied.
    """
    if not isinstance(datasets, dict):
        new_datasets = discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=src_store,
            filter_ktk_cube_dataset_ids=datasets,
        )
    else:
        new_datasets = datasets

    if datasets is None:
        if not new_datasets:
            raise RuntimeError("{} not found in source store".format(cube))
    else:
        unknown_datasets = set(datasets) - set(new_datasets)
        if unknown_datasets:
            raise RuntimeError(
                "{cube}, datasets {datasets} do not exist in source store".format(
                    cube=cube, datasets=unknown_datasets
                )
            )

    existing_datasets = discover_datasets_unchecked(cube.uuid_prefix, tgt_store)

    if not overwrite:
        for ktk_cube_dataset_id in sorted(new_datasets.keys()):
            if ktk_cube_dataset_id in existing_datasets:
                raise RuntimeError(
                    'Dataset "{uuid}" exists in target store but overwrite was set to False'.format(
                        uuid=new_datasets[ktk_cube_dataset_id].uuid
                    )
                )

    all_datasets = copy(existing_datasets)
    all_datasets.update(new_datasets)

    check_datasets(all_datasets, cube)
    return new_datasets


def get_copy_keys(
    cube: Cube,
    src_store: Union[Callable[[], KeyValueStore], KeyValueStore],
    tgt_store: Union[Callable[[], KeyValueStore], KeyValueStore],
    overwrite: bool,
    datasets: Optional[Union[Iterable[str], Dict[str, DatasetMetadata]]] = None,
):
    """
    Get and check keys that should be copied from one store to another.

    Parameters
    ----------
    cube:
        Cube specification.
    src_store:
        Source KV store.
    tgt_store:
        Target KV store.
    overwrite:
        If possibly existing datasets in the target store should be overwritten.
    datasets:
        Datasets to copy, must all be part of the cube. May be either the result of :func:`~kartothek.api.discover.discover_datasets`, an
        iterable of Ktk_cube dataset ID or ``None`` (in which case entire cube will be copied).

    Returns
    -------
    keys: Set[str]
        Set of keys to copy.

    Raises
    ------
    RuntimeError: In case the copy would not pass successfully or if there is no cube in ``src_store``.
    """
    new_datasets = get_datasets_to_copy(
        cube=cube,
        src_store=src_store,
        tgt_store=tgt_store,
        overwrite=overwrite,
        datasets=datasets,
    )

    keys = set()
    for ktk_cube_dataset_id in sorted(new_datasets.keys()):
        ds = new_datasets[ktk_cube_dataset_id]
        keys |= get_dataset_keys(ds)

    return keys
