import warnings
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd

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


@default_docs
@normalize_args
def delete_dataset(dataset_uuid=None, store=None, factory=None):
    """
    Delete the entire dataset from the store.

    Parameters
    ----------
    """

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        load_schema=False,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
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
    tables: Optional[List[str]] = None,
    columns: Dict[str, List[str]] = None,
    concat_partitions_on_primary_index: bool = False,
    predicate_pushdown_to_io: bool = True,
    categoricals: Dict[str, List[str]] = None,
    label_filter: Callable = None,
    dates_as_object: bool = False,
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
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=True,
    )

    mps = read_dataset_as_metapartitions(
        tables=tables,
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
        dispatch_by=dispatch_by,
        dispatch_metadata=False,
    )
    return [mp.data for mp in mps]


@default_docs
def read_dataset_as_metapartitions(
    dataset_uuid=None,
    store=None,
    tables=None,
    columns=None,
    concat_partitions_on_primary_index=False,
    predicate_pushdown_to_io=True,
    categoricals=None,
    label_filter=None,
    dates_as_object=False,
    predicates=None,
    factory=None,
    dispatch_by=None,
    dispatch_metadata=True,
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
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    if len(ds_factory.tables) > 1:
        warnings.warn(
            "Trying to read a dataset with multiple internal tables. This functionality will be removed in the next "
            "major release. If you require a multi tabled data format, we recommend to switch to the kartothek Cube "
            "functionality. "
            "https://kartothek.readthedocs.io/en/stable/guide/cube/kartothek_cubes.html",
            DeprecationWarning,
        )

    from .iter import read_dataset_as_metapartitions__iterator

    ds_iter = read_dataset_as_metapartitions__iterator(
        tables=tables,
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
        dispatch_by=dispatch_by,
        dispatch_metadata=dispatch_metadata,
    )
    return list(ds_iter)


def _check_compatible_list(table, obj, argument_name=""):
    if obj is None:
        return obj
    elif isinstance(obj, dict):
        if table not in obj:
            raise ValueError(
                "Provided table {} is not compatible with input from argument {}.".format(
                    table, argument_name
                )
            )
        return obj
    elif isinstance(obj, list):
        return {table: obj}
    else:
        raise TypeError(
            "Unknown type encountered for argument {}. Expected `list`, got `{}` instead".format(
                argument_name, type(obj)
            )
        )


@default_docs
def read_table(
    dataset_uuid: Optional[str] = None,
    store=None,
    table: Optional[str] = SINGLE_TABLE,
    columns: Dict[str, List[str]] = None,
    concat_partitions_on_primary_index: bool = False,
    predicate_pushdown_to_io: bool = True,
    categoricals: Dict[str, List[str]] = None,
    label_filter: Callable = None,
    dates_as_object: bool = False,
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
    if concat_partitions_on_primary_index is not False:
        warnings.warn(
            "The keyword `concat_partitions_on_primary_index` is deprecated and will be removed in the next major release.",
            DeprecationWarning,
        )

    if not isinstance(table, str):
        raise TypeError("Argument `table` needs to be a string")

    columns = _check_compatible_list(table, columns, "columns")
    categoricals = _check_compatible_list(table, categoricals, "categoricals")

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )
    partitions = read_dataset_as_dataframes(
        tables=[table],
        columns=columns,
        concat_partitions_on_primary_index=concat_partitions_on_primary_index,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        label_filter=label_filter,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
    )

    empty_df = empty_dataframe_from_schema(
        schema=ds_factory.table_meta[table],
        columns=columns[table] if columns is not None else None,
    )
    if categoricals:
        empty_df = empty_df.astype({col: "category" for col in categoricals[table]})
    dfs = [partition_data[table] for partition_data in partitions] + [empty_df]
    # require meta 4 otherwise, can't construct types/columns
    if categoricals:
        dfs = align_categories(dfs, categoricals[table])
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
    output_dataset_uuid: Optional[str] = None,
    delete_scope: Optional[Iterable[Dict[str, Any]]] = None,
    metadata: Dict = None,
    df_serializer: DataFrameSerializer = None,
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
    if output_dataset_uuid is not None:
        warnings.warn(
            "The keyword `output_dataset_uuid` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    if df_serializer is not None:
        warnings.warn(
            "The keyword `df_serializer` is deprecated and will be removed in the next major release.",
            DeprecationWarning,
        )

    if not new_partitions and not metadata and not delete_scope:
        raise ValueError(
            "Need to provide either new data, new metadata or a delete scope. None of it was provided."
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
        new_partitions, metadata_version=metadata_version
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
        if len(mp.files) == 0:
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
    store,
    dataset_uuid,
    dfs,
    metadata=None,
    partition_on=None,
    df_serializer=None,
    overwrite=False,
    secondary_indices=None,
    metadata_storage_format=DEFAULT_METADATA_STORAGE_FORMAT,
    metadata_version=DEFAULT_METADATA_VERSION,
):
    """
    Utility function to store a list of dataframes as a partitioned dataset with multiple tables (files).

    Useful for very small datasets where all data fits into memory.

    Parameters
    ----------
    dfs: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        The dataframe(s) to be stored.

    Returns
    -------
    The stored dataset

    """
    if isinstance(dfs, (pd.DataFrame, dict)):
        dfs = [dfs]
        warnings.warn(
            "Passing a single dataframe instead of an iterable is deprecated and may "
            "be removed in the next major release.",
            DeprecationWarning,
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
    )


@default_docs
@normalize_args
def create_empty_dataset_header(
    store,
    dataset_uuid,
    table_meta,
    partition_on=None,
    metadata=None,
    overwrite=False,
    metadata_storage_format=DEFAULT_METADATA_STORAGE_FORMAT,
    metadata_version=DEFAULT_METADATA_VERSION,
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

    for table, schema in table_meta.items():
        table_meta[table] = make_meta(schema, origin=table, partition_keys=partition_on)
        store_schema_metadata(
            schema=table_meta[table],
            dataset_uuid=dataset_uuid,
            store=store,
            table=table,
        )
    dataset_builder = DatasetMetadataBuilder(
        uuid=dataset_uuid,
        metadata_version=metadata_version,
        partition_keys=partition_on,
        explicit_partitions=False,
        table_meta=table_meta,
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
    store=None,
    dataset_uuid=None,
    data=None,
    metadata=None,
    df_serializer=None,
    overwrite=False,
    metadata_merger=None,
    metadata_version=DEFAULT_METADATA_VERSION,
    partition_on=None,
    factory=None,
    secondary_indices=None,
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
    if metadata is not None:
        warnings.warn(
            "The keyword `metadata` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    if overwrite is not False:
        warnings.warn(
            "The keyword `overwrite` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    if metadata_merger is not None:
        warnings.warn(
            "The keyword `metadata_merger` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    if data is None:
        raise TypeError("The parameter `data` is not optional")
    _, ds_metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=lazy_store(store),
        ds_factory=factory,
        default_metadata_version=metadata_version,
        partition_on=partition_on,
    )

    mp = parse_input_to_metapartition(obj=data, metadata_version=ds_metadata_version)
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
    store=None,
    dataset_uuid: Optional[str] = None,
    delete_scope=None,
    metadata=None,
    df_serializer=None,
    metadata_merger: Callable = None,
    central_partition_metadata: bool = True,
    default_metadata_version=DEFAULT_METADATA_VERSION,
    partition_on: Optional[List[str]] = None,
    load_dynamic_metadata: bool = True,
    sort_partitions_by: Optional[str] = None,
    secondary_indices: Optional[List[str]] = None,
    factory: Optional[DatasetFactory] = None,
) -> DatasetMetadata:
    """
    Update a kartothek dataset in store at once, using a list of dataframes.

    Useful for datasets which do not fit into memory.

    Parameters
    ----------
    df_list: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        The dataframe(s) to be stored.

    Returns
    -------
    The dataset metadata object (:class:`~kartothek.core.dataset.DatasetMetadata`).
    """
    if load_dynamic_metadata is not True:
        warnings.warn(
            "The keyword `load_dynamic_metadata` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    if central_partition_metadata is not True:
        warnings.warn(
            "The keyword `central_partition_metadata` has no use and will be removed in the next major release ",
            DeprecationWarning,
        )

    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        ds_factory=factory,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
    )

    secondary_indices = _ensure_compatible_indices(ds_factory, secondary_indices)

    mp = parse_input_to_metapartition(
        df_list,
        metadata_version=metadata_version,
        expected_secondary_indices=secondary_indices,
    )

    if sort_partitions_by:
        mp = mp.apply(partial(sort_values_categorical, columns=sort_partitions_by))

    if partition_on:
        mp = mp.partition_on(partition_on)

    if secondary_indices:
        mp = mp.build_indices(secondary_indices)

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
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    cols_to_load = {
        table: set(columns) & set(meta.names)
        for table, meta in ds_factory.table_meta.items()
    }
    cols_to_load = {table: cols for table, cols in cols_to_load.items() if cols}

    new_partitions = []
    for mp in dispatch_metapartitions_from_factory(ds_factory):
        mp = mp.load_dataframes(
            store=ds_factory.store,
            tables=list(cols_to_load.keys()),
            columns=cols_to_load,
        )
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
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        load_dataset_metadata=False,
    )

    nested_files = dispatch_files_to_gc(
        dataset_uuid=None, store_factory=None, chunk_size=None, factory=ds_factory
    )

    # Given that `nested_files` is a generator with a single element, just
    # return the output of `delete_files` on that element.
    return delete_files(next(nested_files), store_factory=ds_factory.store_factory)
