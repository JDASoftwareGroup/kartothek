# -*- coding: utf-8 -*-


from functools import partial

from kartothek.core import naming
from kartothek.core.utils import _check_callable
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.docs import default_docs
from kartothek.io_components.metapartition import (
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.utils import normalize_args
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)


def _store_dataset_from_partitions_flat(mpss, *args, **kwargs):
    return store_dataset_from_partitions(
        [mp for sublist in mpss for mp in sublist], *args, **kwargs
    )


@default_docs
@normalize_args
def store_bag_as_dataset(
    bag,
    store,
    dataset_uuid=None,
    metadata=None,
    df_serializer=None,
    overwrite=False,
    metadata_merger=None,
    metadata_version=naming.DEFAULT_METADATA_VERSION,
    partition_on=None,
    metadata_storage_format=naming.DEFAULT_METADATA_STORAGE_FORMAT,
    secondary_indices=None,
):
    """
    Transform and store a dask.bag of dictionaries containing
    dataframes to a kartothek dataset in store.

    This is the dask.bag-equivalent of
    :func:`store_delayed_as_dataset`. See there
    for more detailed documentation on the different possible input types.

    Parameters
    ----------
    bag: dask.bag
        A dask bag containing dictionaries of dataframes or dataframes.

    Returns
    -------
    A dask.delayed dataset object.
    """
    _check_callable(store)
    if dataset_uuid is None:
        dataset_uuid = gen_uuid()

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    input_to_mps = partial(
        parse_input_to_metapartition, metadata_version=metadata_version
    )
    mps = bag.map(input_to_mps)

    if partition_on:
        mps = mps.map(MetaPartition.partition_on, partition_on=partition_on)

    if secondary_indices:
        mps = mps.map(MetaPartition.build_indices, columns=secondary_indices)

    mps = mps.map(
        MetaPartition.store_dataframes,
        store=store,
        df_serializer=df_serializer,
        dataset_uuid=dataset_uuid,
    )

    aggregate = partial(
        _store_dataset_from_partitions_flat,
        dataset_uuid=dataset_uuid,
        store=store,
        dataset_metadata=metadata,
        metadata_merger=metadata_merger,
        metadata_storage_format=metadata_storage_format,
    )

    result = mps.reduction(perpartition=list, aggregate=aggregate, split_every=False)

    return result.to_delayed()
