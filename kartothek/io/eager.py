from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd
from simplekv import KeyValueStore

from kartothek.core.common_metadata import (
    empty_dataframe_from_schema,
    make_meta,
    store_schema_metadata,
)
from kartothek.core.dataset import DatasetMetadata, DatasetMetadataBuilder
from kartothek.core.docs import default_docs
from kartothek.core.factory import DatasetFactory, _ensure_factory
from kartothek.core.naming import (
    DEFAULT_METADATA_STORAGE_FORMAT,
    DEFAULT_METADATA_VERSION,
    PARQUET_FILE_SUFFIX,
    get_partition_file_prefix,
)
from kartothek.core.typing import StoreInput
from kartothek.core.utils import lazy_store
from kartothek.io.iter import store_dataframes_as_dataset__iter
from kartothek.io_components.delete import (
    delete_common_metadata,
    delete_indices,
    delete_top_level_metadata,
)
from kartothek.io_components.gc import delete_files, dispatch_files_to_gc
from kartothek.io_components.index import update_indices_from_partitions
from kartothek.io_components.metapartition import (
    SINGLE_TABLE,
    MetaPartition,
    parse_input_to_metapartition,
)
from kartothek.io_components.read import dispatch_metapartitions_from_factory
from kartothek.io_components.update import update_dataset_from_partitions
from kartothek.io_components.utils import (
    _ensure_compatible_indices,
    align_categories,
    normalize_args,
    sort_values_categorical,
    validate_partition_keys,
)
from kartothek.io_components.write import raise_if_dataset_exists
from kartothek.serialization import DataFrameSerializer

__all__ = (
    "delete_dataset",
    "read_dataset_as_dataframes",
    "read_table",
    "commit_dataset",
    "store_dataframes_as_dataset",
    "create_empty_dataset_header",
    "write_single_partition",
    "update_dataset_from_dataframes",
    "build_dataset_indices",
    "garbage_collect_dataset",
)


@default_docs
@normalize_args
def delete_dataset(dataset_uuid=None, store=None, factory=None):
    """
    Delete the entire dataset from the store.

    Parameters
    ----------
    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, load_schema=False, store=store, factory=factory,
    )

    # Remove possibly unreferenced files
    garbage_collect_dataset(factory=ds_factory)

    # Delete indices first since they do not affect dataset integrity
    delete_indices(dataset_factory=ds_factory)

    for metapartition in dispatch_metapartitions_from_factory(ds_factory):
        metapartition = cast(MetaPartition, metapartition)
        metapartition.delete_from_store(dataset_uuid=dataset_uuid, store=store)

    # delete common metadata after partitions
    delete_common_metadata(dataset_factory=ds_factory)

    # Delete the top level metadata file
    delete_top_level_metadata(dataset_factory=ds_factory)


@default_docs
def read_dataset_as_dataframes(
    dataset_uuid: Optional[str] = None,
    store=None,
    columns: Dict[str, List[str]] = None,
    predicate_pushdown_to_io: bool = True,
    categoricals: List[str] = None,
    dates_as_object: bool = True,
    predicates: Optional[List[List[Tuple[str, str, Any]]]] = None,
    factory: Optional[DatasetFactory] = None,
    dispatch_by: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Read a dataset as a list of dataframes.

    Every element of the list corresponds to a physical partition.

    Parameters
    ----------

    Returns
    -------
    List[pandas.DataFrame]
        Returns a list of pandas.DataFrame. One element per partition

    Examples
    --------
    Dataset in store contains two partitions with two files each

    .. code ::

        >>> import storefact
        >>> from kartothek.io.eager import read_dataset_as_dataframes

        >>> store = storefact.get_store_from_url('s3://bucket_with_dataset')

        >>> dfs = read_dataset_as_dataframes('dataset_uuid', store, 'core')

    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    mps = read_dataset_as_metapartitions(
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
        dispatch_by=dispatch_by,
    )
    return [mp.data for mp in mps]


@default_docs
def read_dataset_as_metapartitions(
    dataset_uuid=None,
    store=None,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals=None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dispatch_by=None,
):
    """
    Read a dataset as a list of :class:`kartothek.io_components.metapartition.MetaPartition`.

    Every element of the list corresponds to a physical partition.

    Parameters
    ----------

    Returns
    -------
    List[kartothek.io_components.metapartition.MetaPartition]
        Returns a tuple of the loaded dataframe and the dataset metadata

    Examples
    --------
    Dataset in store contains two partitions with two files each

    .. code ::

        >>> import storefact
        >>> from kartothek.io.eager import read_dataset_as_dataframe

        >>> store = storefact.get_store_from_url('s3://bucket_with_dataset')

        >>> list_mps = read_dataset_as_metapartitions('dataset_uuid', store, 'core')

    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    from .iter import read_dataset_as_metapartitions__iterator

    ds_iter = read_dataset_as_metapartitions__iterator(
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
        dispatch_by=dispatch_by,
    )
    return list(ds_iter)


@default_docs
def read_table(
    dataset_uuid: Optional[str] = None,
    store=None,
    columns: Dict[str, List[str]] = None,
    predicate_pushdown_to_io: bool = True,
    categoricals: List[str] = None,
    dates_as_object: bool = True,
    predicates: Optional[List[List[Tuple[str, str, Any]]]] = None,
    factory: Optional[DatasetFactory] = None,
) -> pd.DataFrame:
    """
    A utility function to load a single table with multiple partitions as a single dataframe in one go.
    Mostly useful for smaller tables or datasets where all partitions fit into memory.

    The order of partitions is not guaranteed to be stable in the resulting dataframe.

    Parameters
    ----------

    Returns
    -------
    pandas.DataFrame
        Returns a pandas.DataFrame holding the data of the requested columns

    Examples
    --------
    Dataset in store contains two partitions with two files each

    .. code ::

        >>> import storefact
        >>> from kartothek.io.eager import read_table

        >>> store = storefact.get_store_from_url('s3://bucket_with_dataset')

        >>> df = read_table(store, 'dataset_uuid', 'core')

    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )
    partitions = read_dataset_as_dataframes(
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
    )

    empty_df = empty_dataframe_from_schema(schema=ds_factory.schema, columns=columns,)
    if categoricals:
        empty_df = empty_df.astype({col: "category" for col in categoricals})
    dfs = [partition_data for partition_data in partitions] + [empty_df]
    # require meta 4 otherwise, can't construct types/columns
    if categoricals:
        dfs = align_categories(dfs, categoricals)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    # ensure column order
    if len(empty_df.columns) > 0 and list(empty_df.columns) != list(df.columns):
        df = df.reindex(empty_df.columns, copy=False, axis=1)

    return df


@default_docs
@normalize_args
def commit_dataset(
    store: Optional[StoreInput] = None,
    dataset_uuid: Optional[str] = None,
    new_partitions: Optional[Iterable[MetaPartition]] = None,
    delete_scope: Optional[Iterable[Dict[str, Any]]] = None,
    metadata: Dict = None,
    metadata_merger: Callable[[List[Dict]], Dict] = None,
    default_metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: Optional[Iterable[str]] = None,
    factory: Optional[DatasetFactory] = None,
    secondary_indices: Optional[Iterable[str]] = None,
):
    """
    Commit new state to an existing dataset. This can be used for three distinct operations

    1. Add previously written partitions to this dataset

        If for some reasons, the existing pipelines are not sufficient but you need more control, you can write the files outside of a kartothek pipeline and commit them whenever you choose to.

        This should be used in combination with
        :func:`~kartothek.io.eager.write_single_partition` and :func:`~kartothek.io.eager.create_empty_dataset_header`.

        .. code::

            import pandas as pd
            from kartothek.io.eager import write_single_partition, commit_dataset

            store = "hfs://my_store"

            # The partition writing can be done concurrently and distributed if wanted.
            # Only the information about what partitions have been written is required for the commit.
            new_partitions = [
                write_single_partition(
                    store=store,
                    dataset_uuid='dataset_uuid',
                    data=pd.DataFrame({'column': [1, 2]}),
                )
            ]

            new_dataset = commit_dataset(
                store=store,
                dataset_uuid='dataset_uuid',
                new_partitions=new_partitions,
            )

    2. Simple delete of partitions

        If you want to remove some partitions this is one of the simples ways of doing so. By simply providing a delete_scope, this removes the references to these files in an atomic commit.

        .. code::

            commit_dataset(
                store=store,
                dataset_uuid='dataset_uuid',
                delete_scope=[
                    {
                        "partition_column": "part_value_to_be_removed"
                    }
                ],
            )

    3. Add additional metadata

        To add new metadata to an existing dataset

        .. code::

            commit_dataset(
                store=store,
                dataset_uuid='dataset_uuid',
                metadata={"new": "user_metadata"},
            )

        Note::

            If you do not want the new metadata to be merged with the existing one, povide a custom ``metadata_merger``

    Parameters
    ----------
    new_partitions:
        Input partition to be committed.

    """

    if not new_partitions and not metadata and not delete_scope:
        raise ValueError(
            "Need to provide either new data, new metadata or a delete scope. None of it was provided."
        )
    if new_partitions:
        tables_in_partitions = {mp.table_name for mp in new_partitions}
        if len(tables_in_partitions) > 1:
            raise RuntimeError(
                f"Cannot commit more than one table to a dataset but got tables {sorted(tables_in_partitions)}"
            )
    store = lazy_store(store)
    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        ds_factory=factory,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
    )

    mps = parse_input_to_metapartition(
        new_partitions,
        metadata_version=metadata_version,
        table_name=ds_factory.table_name,
    )

    if secondary_indices:
        mps = mps.build_indices(columns=secondary_indices)

    mps_list = [_maybe_infer_files_attribute(mp, dataset_uuid) for mp in mps]

    dmd = update_dataset_from_partitions(
        mps_list,
        store_factory=store,
        dataset_uuid=dataset_uuid,
        ds_factory=ds_factory,
        delete_scope=delete_scope,
        metadata=metadata,
        metadata_merger=metadata_merger,
    )
    return dmd


def _maybe_infer_files_attribute(metapartition, dataset_uuid):
    new_mp = metapartition.as_sentinel()
    for mp in metapartition:
        if mp.file is None:
            if mp.data is None or len(mp.data) == 0:
                raise ValueError(
                    "Trying to commit partitions without `data` or `files` information."
                    "Either one is necessary to infer the dataset tables"
                )
            new_files = {}
            for table in mp.data:
                new_files[table] = (
                    get_partition_file_prefix(
                        dataset_uuid=dataset_uuid,
                        partition_label=mp.label,
                        table=table,
                        metadata_version=mp.metadata_version,
                    )
                    + PARQUET_FILE_SUFFIX  # noqa: W503 line break before binary operator
                )
            mp = mp.copy(files=new_files)

        new_mp = new_mp.add_metapartition(mp)
    return new_mp


@default_docs
@normalize_args
def store_dataframes_as_dataset(
    store: KeyValueStore,
    dataset_uuid: str,
    dfs: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    partition_on: Optional[List[str]] = None,
    df_serializer: Optional[DataFrameSerializer] = None,
    overwrite: bool = False,
    secondary_indices=None,
    metadata_storage_format=DEFAULT_METADATA_STORAGE_FORMAT,
    metadata_version=DEFAULT_METADATA_VERSION,
    table_name: str = SINGLE_TABLE,
):
    """
    Utility function to store a list of dataframes as a partitioned dataset with multiple tables (files).

    Useful for very small datasets where all data fits into memory.

    Parameters
    ----------
    dfs:
        The dataframe(s) to be stored.

    """
    if isinstance(dfs, pd.DataFrame):
        raise TypeError(
            f"Please pass a list of pandas.DataFrame as input. Instead got {type(dfs)}"
        )

    return store_dataframes_as_dataset__iter(
        dfs,
        store=store,
        dataset_uuid=dataset_uuid,
        metadata=metadata,
        partition_on=partition_on,
        df_serializer=df_serializer,
        overwrite=overwrite,
        secondary_indices=secondary_indices,
        metadata_storage_format=metadata_storage_format,
        metadata_version=metadata_version,
        table_name=table_name,
    )


@default_docs
@normalize_args
def create_empty_dataset_header(
    store,
    dataset_uuid,
    schema,
    partition_on=None,
    metadata=None,
    overwrite=False,
    metadata_storage_format=DEFAULT_METADATA_STORAGE_FORMAT,
    metadata_version=DEFAULT_METADATA_VERSION,
    table_name: str = SINGLE_TABLE,
):
    """
    Create an dataset header without any partitions. This may be used in combination
    with :func:`~kartothek.io.eager.write_single_partition` to create implicitly partitioned datasets.

    .. note::

        The created dataset will **always** have explicit_partition==False

    .. warning::

        This function should only be used in very rare occasions. Usually you're better off using
        full end-to-end pipelines.

    Parameters
    ----------
    """
    store = lazy_store(store)()
    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    schema = make_meta(schema, origin=table_name, partition_keys=partition_on)
    store_schema_metadata(
        schema=schema, dataset_uuid=dataset_uuid, store=store, table=table_name,
    )
    dataset_builder = DatasetMetadataBuilder(
        uuid=dataset_uuid,
        metadata_version=metadata_version,
        partition_keys=partition_on,
        explicit_partitions=False,
        schema=schema,
    )
    if metadata:
        for key, value in metadata.items():
            dataset_builder.add_metadata(key, value)
    if metadata_storage_format.lower() == "json":
        store.put(*dataset_builder.to_json())
    elif metadata_storage_format.lower() == "msgpack":
        store.put(*dataset_builder.to_msgpack())
    else:
        raise ValueError(
            "Unknown metadata storage format encountered: {}".format(
                metadata_storage_format
            )
        )
    return dataset_builder.to_dataset()


@default_docs
@normalize_args
def write_single_partition(
    store: Optional[KeyValueStore] = None,
    dataset_uuid: Optional[str] = None,
    data=None,
    df_serializer: Optional[DataFrameSerializer] = None,
    metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: Optional[List[str]] = None,
    factory=None,
    secondary_indices=None,
    table_name: str = SINGLE_TABLE,
):
    """
    Write the parquet file(s) for a single partition. This will **not** update the dataset header and can therefore
    be used for highly concurrent dataset writes.

    For datasets with explicit partitions, the dataset header can be updated by calling
    :func:`kartothek.io.eager.commit_dataset` with the output of this function.

    .. note::

        It is highly recommended to use the full pipelines whenever possible. This functionality should be
        used with caution and should only be necessary in cases where traditional pipeline scheduling is not an
        option.

    .. note::

        This function requires an existing dataset metadata file and the schemas for the tables to be present.
        Either you have ensured that the dataset always exists though some other means or use
        :func:`create_empty_dataset_header` at the start of your computation to ensure the basic dataset
        metadata is there.

    Parameters
    ----------
    data: Dict
        The input is defined according to :func:`~kartothek.io_components.metapartition.parse_input_to_metapartition`

    Returns
    -------
    An empty :class:`~kartothek.io_components.metapartition.MetaPartition` referencing the new files
    """
    if data is None:
        raise TypeError("The parameter `data` is not optional")
    dataset_factory, ds_metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=lazy_store(store),
        ds_factory=factory,
        default_metadata_version=metadata_version,
        partition_on=partition_on,
    )
    if dataset_factory.table_name:
        if dataset_factory.table_name != table_name:
            raise RuntimeError(
                f"Trying to write a partition with table name {table_name} but dataset {dataset_factory.dataset_uuid} has already table {dataset_factory.table_name}."
            )
    mp = parse_input_to_metapartition(
        obj=data, metadata_version=ds_metadata_version, table_name=table_name
    )

    if partition_on:
        mp = mp.partition_on(partition_on)

    if secondary_indices:
        mp = mp.build_indices(columns=secondary_indices)

    mp = mp.validate_schema_compatible(dataset_uuid=dataset_uuid, store=store)

    mp = mp.store_dataframes(
        store=store, dataset_uuid=dataset_uuid, df_serializer=df_serializer
    )
    return mp


@default_docs
@normalize_args
def update_dataset_from_dataframes(
    df_list: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    store: Optional[KeyValueStore] = None,
    dataset_uuid: Optional[str] = None,
    delete_scope=None,
    metadata=None,
    df_serializer: Optional[DataFrameSerializer] = None,
    metadata_merger: Callable = None,
    default_metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: Optional[List[str]] = None,
    sort_partitions_by: Optional[str] = None,
    secondary_indices: Optional[List[str]] = None,
    table_name: str = SINGLE_TABLE,
    factory: Optional[DatasetFactory] = None,
) -> DatasetMetadata:
    """
    Update a kartothek dataset in store at once, using a list of dataframes.

    Useful for datasets which do not fit into memory.

    Parameters
    ----------
    df_list:
        The dataframe(s) to be stored.

    Returns
    -------
    The dataset metadata object (:class:`~kartothek.core.dataset.DatasetMetadata`).

    See Also
    --------
    :ref:`mutating_datasets`
    """
    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        ds_factory=factory,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
    )

    # ensured by normalize_args but mypy doesn't recognize it
    assert secondary_indices is not None

    inferred_indices = _ensure_compatible_indices(ds_factory, secondary_indices)
    del secondary_indices

    mp = parse_input_to_metapartition(
        df_list, metadata_version=metadata_version, table_name=table_name,
    )

    if sort_partitions_by:
        mp = mp.apply(partial(sort_values_categorical, columns=sort_partitions_by))

    if partition_on:
        mp = mp.partition_on(partition_on)

    if inferred_indices:
        mp = mp.build_indices(inferred_indices)

    mp = mp.store_dataframes(
        store=store, dataset_uuid=dataset_uuid, df_serializer=df_serializer
    )

    return update_dataset_from_partitions(
        mp,
        store_factory=store,
        dataset_uuid=dataset_uuid,
        ds_factory=ds_factory,
        delete_scope=delete_scope,
        metadata=metadata,
        metadata_merger=metadata_merger,
    )


@default_docs
@normalize_args
def build_dataset_indices(store, dataset_uuid, columns, factory=None):
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

    cols_to_load = set(columns) & set(ds_factory.schema.names)

    new_partitions = []
    for mp in dispatch_metapartitions_from_factory(ds_factory):
        mp = mp.load_dataframes(store=ds_factory.store, columns=cols_to_load,)
        mp = mp.build_indices(columns=columns)
        mp = mp.remove_dataframes()  # Remove dataframe from memory
        new_partitions.append(mp)

    return update_indices_from_partitions(
        new_partitions, dataset_metadata_factory=ds_factory
    )


@default_docs
@normalize_args
def garbage_collect_dataset(dataset_uuid=None, store=None, factory=None):
    """
    Remove auxiliary files that are no longer tracked by the dataset.

    These files include indices that are no longer referenced by the metadata
    as well as files in the directories of the tables that are no longer
    referenced. The latter is only applied to static datasets.

    Parameters
    ----------
    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid, store=store, factory=factory,
    )

    nested_files = dispatch_files_to_gc(
        dataset_uuid=None, store_factory=None, chunk_size=None, factory=ds_factory
    )

    # Given that `nested_files` is a generator with a single element, just
    # return the output of `delete_files` on that element.
    return delete_files(next(nested_files), store_factory=ds_factory.store_factory)
