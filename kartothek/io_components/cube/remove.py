from functools import reduce

from kartothek.core.cube.conditions import Conjunction
from kartothek.core.cube.constants import KTK_CUBE_METADATA_VERSION
from kartothek.io_components.metapartition import MetaPartition
from kartothek.utils.converters import converter_str_set_optional
from kartothek.utils.ktk_adapters import get_partition_dataframe

__all__ = ("prepare_metapartitions_for_removal_action",)


def prepare_metapartitions_for_removal_action(
    cube, store, conditions, ktk_cube_dataset_ids, existing_datasets
):
    """
    Prepare MetaPartition to express removal of given data range from cube.

    The MetaPartition must still be written using ``mp.store_dataframes(...)`` and added to the Dataset using a
    kartothek update method.

    Parameters
    ----------
    cube: kartothek.core.cube.cube.Cube
        Cube spec.
    store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Store.
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied, optional. Defaults to "entire cube".
    ktk_cube_dataset_ids: Optional[Union[Iterable[Union[Str, Bytes]], Union[Str, Bytes]]]
        Ktk_cube dataset IDs to apply the remove action to, optional. Default to "all".
    existing_datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Existing datasets.

    Returns
    -------
    metapartitions: Dict[str, Tuple[kartothek.core.dataset.DatasetMetadata,
            kartothek.io_components.metapartition.MetaPartition, List[Dict[str, Any]]]]
        MetaPartitions that should be written and updatet to the kartothek datasets as well as the ``delete_scope`` for
        kartothek.
    """
    conditions = Conjunction(conditions)
    conditions_split = conditions.split_by_column()
    if set(conditions_split.keys()) - set(cube.partition_columns):
        raise ValueError(
            "Can only remove partitions with conditions concerning cubes physical partition columns."
        )

    ktk_cube_dataset_ids = converter_str_set_optional(ktk_cube_dataset_ids)
    if ktk_cube_dataset_ids is not None:
        unknown_dataset_ids = ktk_cube_dataset_ids - set(existing_datasets.keys())
        if unknown_dataset_ids:
            raise ValueError(
                "Unknown ktk_cube_dataset_ids: {}".format(
                    ", ".join(sorted(unknown_dataset_ids))
                )
            )
    else:
        ktk_cube_dataset_ids = set(existing_datasets.keys())

    metapartitions = {}
    for ktk_cube_dataset_id in ktk_cube_dataset_ids:
        ds = existing_datasets[ktk_cube_dataset_id]
        ds = ds.load_partition_indices()
        mp = _prepare_mp_empty(ds)

        if not ds.partition_keys:
            # no partition keys --> delete all
            delete_scope = [{}]
        else:

            df_partitions = get_partition_dataframe(dataset=ds, cube=cube)
            df_partitions = df_partitions.drop_duplicates()
            local_condition = reduce(
                lambda a, b: a & b,
                (
                    cond
                    for col, cond in conditions_split.items()
                    if col in df_partitions.columns
                ),
                Conjunction([]),
            )
            df_partitions = local_condition.filter_df(df_partitions)

            delete_scope = df_partitions.to_dict(orient="records")

        metapartitions[ktk_cube_dataset_id] = (ds, mp, delete_scope)

    return metapartitions


def _prepare_mp_empty(dataset):
    """
    Generate empty partition w/o any data for given cube.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        Dataset to build empty MetaPartition for.

    Returns
    -------
    mp: kartothek.io_components.metapartition.MetaPartition
        MetaPartition, must still be added to the Dataset using a kartothek update method.
    """
    return MetaPartition(
        label=None,
        metadata_version=KTK_CUBE_METADATA_VERSION,
        partition_keys=dataset.partition_keys,
    )
