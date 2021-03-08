from __future__ import absolute_import

from copy import copy

from kartothek.api.discover import check_datasets, discover_datasets_unchecked
from kartothek.core.dataset import DatasetMetadataBuilder
from kartothek.utils.ktk_adapters import get_dataset_keys

__all__ = ("get_copy_keys",)


def _assert_datasets_contained_in(cube, ds_needles, ds_haystack):
    unknown_datasets = set(ds_needles) - set(ds_haystack)
    if unknown_datasets:
        raise RuntimeError(
            "{cube}, datasets {datasets} do not exist in source store".format(
                cube=cube, datasets=unknown_datasets
            )
        )


def get_copy_keys(
    cube, src_store, tgt_store, overwrite, datasets=None, datasets_to_rename=None
):
    """
    Get and check keys that should be copied from one store to another.

    Parameters
    ----------
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    src_store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        Source KV store.
    tgt_store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        Target KV store.
    overwrite: bool
        If possibly existing datasets in the target store should be overwritten.
    datasets: Union[None, Iterable[str], Dict[str, kartothek.core.dataset.DatasetMetadata]]
        Datasets to copy, must all be part of the cube. May be either the result of :func:`~kartothek.api.discover.discover_datasets`, an
        iterable of Ktk_cube dataset ID or ``None`` (in which case entire cube will be copied).
    datasets_to_rename: Optional[Dict[str, str]]
        Optional dict of {old dataset name:  new dataset name} entries. If specified,
        the corresponding datasets will be renamed accordingly. Unknown keys will cause
        an error.

    Returns
    -------
    keys: Set[str]
        Set of keys to copy.

    Raises
    ------
    RuntimeError: In case the copy would not pass successfully or if there is no cube in ``src_store``.
    """

    # if datasets is a dict, i.e. the result of discover_datasets(): use it.
    # otherwise, call discover_datasets_unchecked() to create such a dict
    if not isinstance(datasets, dict):
        datasets_to_copy = discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=src_store,
            filter_ktk_cube_dataset_ids=datasets,
        )
    else:
        datasets_to_copy = datasets

    if datasets is None:
        if not datasets_to_copy:
            raise RuntimeError("{} not found in source store".format(cube))
    else:
        # check if datasets parameter contains unknown, i.e. non-existing, datasets.
        # this may only happen if datasets is a list of dataset names.
        _assert_datasets_contained_in(cube, datasets, datasets_to_copy)

    if datasets_to_rename:
        # if datasets shall be renamed: check if the list of datasets to rename is a
        # subset of the datasets to copy
        _assert_datasets_contained_in(cube, datasets_to_rename, datasets_to_copy)
    else:
        datasets_to_rename = {}

    existing_datasets = discover_datasets_unchecked(cube.uuid_prefix, tgt_store)

    if not overwrite:
        # If no target data shall be overwritten, check if the selected datasets do
        # already exist in the target store. Note that the dataset name may change.
        for ktk_cube_dataset_id in sorted(datasets_to_copy.keys()):
            ktk_target_dataset_id = datasets_to_rename.get(
                ktk_cube_dataset_id, ktk_cube_dataset_id
            )
            if ktk_target_dataset_id in existing_datasets:
                raise RuntimeError(
                    'Dataset "{uuid}" exists in target store but overwrite was set to False'.format(
                        uuid=existing_datasets[ktk_target_dataset_id].uuid
                    )
                )

    all_datasets = copy(existing_datasets)
    for ds, ds_meta in datasets_to_copy.items():
        renamed_ds = datasets_to_rename.get(ds, ds)
        all_datasets[renamed_ds] = (
            DatasetMetadataBuilder.from_dataset(ds_meta)
            .modify_dataset_name(renamed_ds)
            .to_dataset()
        )

    check_datasets(all_datasets, cube)

    keys = set()
    for ktk_cube_dataset_id in sorted(datasets_to_copy.keys()):
        ds = datasets_to_copy[ktk_cube_dataset_id]
        keys |= get_dataset_keys(ds)

    return keys
