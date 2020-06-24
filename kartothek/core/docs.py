"""
This is a helper module to simplify code documentation
"""

import inspect
from io import StringIO

_PARAMETER_MAPPING = {
    "store": """
    store: callable
        Factory function producing a KeyValueStore.""",
    "overwrite": """
    overwrite: bool, optional
        If True, allow overwrite of an existing dataset.""",
    "label_merger": """
    label_merger: callable, optional
        By default the shorter label of either the left or right partition is chosen
        as the merged partition label. Supplying a callable here, allows you to override
        the default behavior and create a new label from all input labels
        (depending on the matches this might be more than two values)""",
    "metadata_merger": """
    metadata_merger: callable, optional
         By default partition metadata is combined using the :func:`~kartothek.io_components.utils.combine_metadata` function.
         You can supply a callable here that implements a custom merge operation on the metadata dictionaries
         (depending on the matches this might be more than two values).""",
    "table": """
    table: str, optional
        The table to be loaded. If none is specified, the default 'table' is used.""",
    "tables": """
    tables : list of str
        A list of tables to be loaded. If None is given, all tables of
        a partition are loaded""",
    "table_meta": """
    table_meta: Dict[str, SchemaWrapper]
        The dataset table schemas""",
    "columns": """
    columns : dict of list of string, optional
        A dictionary mapping tables to list of columns. Only the specified
        columns are loaded for the corresponding table. If a specfied table or column is
        not present in the dataset, a ValueError is raised.""",
    "dispatch_by": """
    dispatch_by: list of strings, optional
        List of index columns to group and partition the jobs by.
        There will be one job created for every observed index value
        combination. This may result in either many very small partitions or in
        few very large partitions, depending on the index you are using this on.

        .. admonition:: Secondary indices

            This is also useable in combination with secondary indices where
            the physical file layout may not be aligned with the logically
            requested layout. For optimal performance it is recommended to use
            this for columns which can benefit from predicate pushdown since
            the jobs will fetch their data individually and will *not* shuffle
            data in memory / over network.""",
    "df_serializer": """
    df_serializer : DataFrameSerializer, optional
        A pandas DataFrame serialiser from `kartothek.serialization`""",
    "output_dataset_uuid": """
    output_dataset_uuid: basestring, optional
        UUID of the newly created dataset""",
    "output_dataset_metadata": """
    output_dataset_metadata: dict, optional
        Metadata for the merged target dataset. Will be updated with a
        `merge_datasets__pipeline` key that contains the source dataset uuids for
        the merge.""",
    "output_store": """
    output_store : callable
        Factory function producing a KeyValueStore.
        If given, the resulting dataset is written to this store. By default
        the input store""",
    "metadata": """
    metadata : dict, optional
        A dictionary used to update the dataset metadata.""",
    "dataset_uuid": """
    dataset_uuid: str
        The dataset UUID""",
    "metadata_version": """
    metadata_version: int, optional
        The dataset metadata version""",
    "partition_on": """
    partition_on: list
        Column names by which the dataset should be partitioned by physically.
        These columns may later on be used as an Index to improve query performance.
        Partition columns need to be present in all dataset tables.
        Sensitive to ordering.""",
    "predicate_pushdown_to_io": """
    predicate_pushdown_to_io: bool
        Push predicates through to the I/O layer, default True. Disable
        this if you see problems with predicate pushdown for the given
        file even if the file format supports it. Note that this option
        only hides problems in the storage layer that need to be addressed
        there.""",
    "delete_scope": """
    delete_scope: list of dicts
        This defines which partitions are replaced with the input and therefore
        get deleted. It is a lists of query filters for the dataframe in the
        form of a dictionary, e.g.: `[{'column_1': 'value_1'}, {'column_1': 'value_2'}].
        Each query filter will be given to: func: `dataset.query` and the returned
        partitions will be deleted. If no scope is given nothing will be deleted.
        For `kartothek.io.dask.update.update_dataset.*` a delayed object resolving to
        a list of dicts is also accepted.""",
    "categoricals": """
    categoricals : dicts of list of string
        A dictionary mapping tables to list of columns that should be
        loaded as `category` dtype instead of the inferred one.""",
    "label_filter": """
    label_filter: callable
        A callable taking a partition label as a parameter and returns a boolean. The callable will be applied
        to the list of partitions during dispatch and will filter out all partitions for which the callable
        evaluates to False.""",
    "dates_as_object": """
    dates_as_object: bool
        Load pyarrow.date{32,64} columns as ``object`` columns in Pandas
        instead of using ``np.datetime64`` to preserve their type. While
        this improves type-safety, this comes at a performance cost.""",
    "predicates": """
    predicates: list of list of tuple[str, str, Any]
        Optional list of predicates, like [[('x', '>', 0), ...], that are used
        to filter the resulting DataFrame, possibly using predicate pushdown,
        if supported by the file format.
        This parameter is not compatible with filter_query.

        Predicates are expressed in disjunctive normal form (DNF). This means
        that the innermost tuple describes a single column predicate. These
        inner predicates are all combined with a conjunction (AND) into a
        larger predicate. The most outer list then combines all predicates
        with a disjunction (OR). By this, we should be able to express all
        kinds of predicates that are possible using boolean logic.

        Available operators are: `==`, `!=`, `<=`, `>=`, `<`, `>` and `in`.

        .. admonition:: Categorical data

            When using order sensitive operators on categorical data we will
            assume that the categories obey a lexicographical ordering.
            This filtering may result in less than optimal performance and may
            be slower than the evaluation on non-categorical data.""",
    "secondary_indices": """
    secondary_indices: List[str]
        A list of columns for which a secondary index should be calculated.""",
    "sort_partitions_by": """
    sort_partitions_by: str
        Provide a column after which the data should be sorted before storage to enable predicate pushdown.""",
    "factory": """
    factory: kartothek.core.factory.DatasetFactory
        A DatasetFactory holding the store and UUID to the source dataset.""",
    "partition_size": """
    partition_size: Optional[int]
        Dask bag partition size. Use a larger numbers to decrease scheduler load and overhead, use smaller numbers for a
        fine-grained scheduling and better resilience against worker errors.""",
    "metadata_storage_format": """
    metadata_storage_format: str
        Optional list of datastorage format to use. Currently supported is `.json` & `.msgpack.zstd"`""",
    "df_generator": """
    df_generator: Iterable[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        The dataframe(s) to be stored""",
    "central_partition_metadata": """
    central_partition_metadata: bool
        This has no use and will be removed in future releases""",
    "default_metadata_version": """
    default_metadata_version: int
        Default metadata version. (Note: Metadata version greater than 3 are only supported)""",
    "load_dynamic_metadata": """
    load_dynamic_metadata: bool
        The keyword `load_dynamic_metadata` is deprecated and will be removed in the next major release.""",
    "concat_partitions_on_primary_index": """
    concat_partitions_on_primary_index: bool
        Concatenate partition based on their primary index values.""",
    "delayed_tasks": """
    delayed_tasks: list of dask.delayed
        Every delayed object represents a partition and should be accepted by
        :func:`~kartothek.io_components.metapartition.parse_input_to_metapartition`""",
    "load_dataset_metadata": """
    load_dataset_metadata: bool
        Optional argument on whether to load the metadata or not""",
    "dispatch_metadata": """
    dispatch_metadata:
        If True, attach dataset user metadata and dataset index information to
        the MetaPartition instances generated.
        Note: This feature is deprecated and this feature toggle is only
        introduced to allow for easier transition.""",
}


def default_docs(func):
    """
    A decorator which automatically takes care of default parameter
    documentation for common pipeline factory parameters
    """
    # TODO (Kshitij68) Bug: The parameters are not come in the same order as listed in the function. For example in `store_dataframes_as_dataset`
    docs = func.__doc__
    new_docs = ""
    signature = inspect.signature(func)

    try:
        buf = StringIO(docs)
        line = True
        while line:
            line = buf.readline()

            if "Parameters" in line:
                indentation_level = len(line) - len(line.lstrip())
                artificial_param_docs = [line, buf.readline()]
                # Include the `-----` line
                for param in signature.parameters.keys():
                    doc = _PARAMETER_MAPPING.get(param, None)
                    if doc and param + ":" not in docs:
                        if not doc.endswith("\n"):
                            doc += "\n"
                        if doc.startswith("\n"):
                            doc = doc[1:]
                        doc_indentation_level = len(doc) - len(doc.lstrip())
                        whitespaces_to_add = indentation_level - doc_indentation_level
                        if whitespaces_to_add < 0:
                            raise RuntimeError(
                                f"Indentation detection went wrong for parameter {param}"
                            )
                        # Adjust the indentation dynamically
                        whitespaces = " " * whitespaces_to_add
                        doc = whitespaces + doc
                        doc = doc.replace("\n", "\n" + whitespaces).rstrip() + "\n"
                        # We are checking if the entire docstring associated with the function is present or not
                        if doc not in docs:
                            artificial_param_docs.append(doc)
                new_docs += "".join(artificial_param_docs)
                continue
            new_docs = "".join([new_docs, line])
        func.__doc__ = new_docs
    except Exception as ex:
        func.__doc__ = str(ex)
    return func
