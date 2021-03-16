from functools import partial
from typing import Optional, Sequence

import dask.bag as db
from dask.delayed import Delayed

from kartothek.core import naming
from kartothek.core.docs import default_docs
from kartothek.core.factory import DatasetFactory, _ensure_factory
from kartothek.core.typing import StoreInput
from kartothek.core.utils import lazy_store
from kartothek.core.uuid import gen_uuid
from kartothek.io.dask._utils import (
    _cast_categorical_to_index_cat,
    _get_data,
    _maybe_get_categoricals_from_index,
)
from kartothek.io_components.index import update_indices_from_partitions
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.utils import normalize_args, raise_if_indices_overlap
from kartothek.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)

__all__ = (
    "read_dataset_as_dataframe_bag",
    "store_bag_as_dataset",
    "build_dataset_indices__bag",
)


def _store_dataset_from_partitions_flat(mpss, *args, **kwargs):
    return store_dataset_from_partitions(
        [mp for sublist in mpss for mp in sublist], *args, **kwargs
    )


def _load_and_concat_metapartitions_inner(mps, *args, **kwargs):
    return MetaPartition.concat_metapartitions(
        [mp.load_dataframes(*args, **kwargs) for mp in mps]
    )


@default_docs
def read_dataset_as_metapartitions_bag(
    dataset_uuid=None,
    store=None,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals=None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dispatch_by=None,
    partition_size=None,
):
    """
    Retrieve dataset as `dask.bag.Bag` of `MetaPartition` objects.

    Parameters
    ----------

    Returns
    -------
    dask.bag.Bag:
        A dask.bag object containing the metapartions.
    """
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    store = ds_factory.store_factory
    mps = dispatch_metapartitions_from_factory(
        dataset_factory=ds_factory, predicates=predicates, dispatch_by=dispatch_by,
    )
    mp_bag = db.from_sequence(mps, partition_size=partition_size)

    if dispatch_by is not None:
        mp_bag = mp_bag.map(
            _load_and_concat_metapartitions_inner,
            store=store,
            columns=columns,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )
    else:
        mp_bag = mp_bag.map(
            MetaPartition.load_dataframes,
            store=store,
            columns=columns,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )

    categoricals_from_index = _maybe_get_categoricals_from_index(
        ds_factory, categoricals
    )

    if categoricals_from_index:

        mp_bag = mp_bag.map(
            MetaPartition.apply,
            func=partial(
                _cast_categorical_to_index_cat, categories=categoricals_from_index
            ),
            type_safe=True,
        )
    return mp_bag


@default_docs
def read_dataset_as_dataframe_bag(
    dataset_uuid=None,
    store=None,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals=None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dispatch_by=None,
    partition_size=None,
):
    """
    Retrieve data as dataframe from a :class:`dask.bag.Bag` of `MetaPartition` objects

    Parameters
    ----------

    Returns
    -------
    dask.bag.Bag
        A dask.bag.Bag which contains the metapartitions and mapped to a function for retrieving the data.
    """
    mps = read_dataset_as_metapartitions_bag(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        dispatch_by=dispatch_by,
        partition_size=partition_size,
    )
    return mps.map(_get_data)


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
    table_name: str = SINGLE_TABLE,
):
    """
    Transform and store a dask.bag of dictionaries containing
    dataframes to a kartothek dataset in store.

    This is the dask.bag-equivalent of
    :func:`~kartothek.io.dask.delayed.store_delayed_as_dataset`. See there
    for more detailed documentation on the different possible input types.

    Parameters
    ----------
    bag: dask.bag.Bag
        A dask bag containing dictionaries of dataframes or dataframes.

    """
    store = lazy_store(store)
    if dataset_uuid is None:
        dataset_uuid = gen_uuid()

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    raise_if_indices_overlap(partition_on, secondary_indices)

    input_to_mps = partial(
        parse_input_to_metapartition,
        metadata_version=metadata_version,
        table_name=table_name,
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

    return mps.reduction(perpartition=list, aggregate=aggregate, split_every=False)


@default_docs
def build_dataset_indices__bag(
    store: Optional[StoreInput],
    dataset_uuid: Optional[str],
    columns: Sequence[str],
    partition_size: Optional[int] = None,
    factory: Optional[DatasetFactory] = None,
) -> Delayed:
    """
    Function which builds a :class:`~kartothek.core.index.ExplicitSecondaryIndex`.

    This function loads the dataset, computes the requested indices and writes
    the indices to the dataset. The dataset partitions itself are not mutated.

    Parameters
    ----------

    """
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    assert ds_factory.schema is not None
    cols_to_load = set(columns) & set(ds_factory.schema.names)

    mps = dispatch_metapartitions_from_factory(ds_factory)

    return (
        db.from_sequence(seq=mps, partition_size=partition_size)
        .map(
            MetaPartition.load_dataframes,
            store=ds_factory.store_factory,
            columns=cols_to_load,
        )
        .map(MetaPartition.build_indices, columns=columns)
        .map(MetaPartition.remove_dataframes)
        .reduction(list, list, split_every=False, out_type=db.Bag)
        .flatten()
        .map_partitions(list)
        .map_partitions(
            update_indices_from_partitions, dataset_metadata_factory=ds_factory
        )
    )
