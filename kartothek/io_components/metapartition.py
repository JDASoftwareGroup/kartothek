import inspect
import io
import logging
import os
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import pyarrow as pa
from simplekv import KeyValueStore

from kartothek.core import naming
from kartothek.core.common_metadata import (
    SchemaWrapper,
    make_meta,
    normalize_column_order,
    read_schema_metadata,
    validate_compatible,
)
from kartothek.core.docs import default_docs
from kartothek.core.index import ExplicitSecondaryIndex, IndexBase
from kartothek.core.index import merge_indices as merge_indices_algo
from kartothek.core.naming import get_partition_file_prefix
from kartothek.core.partition import Partition
from kartothek.core.typing import StoreInput
from kartothek.core.urlencode import decode_key, quote_indices
from kartothek.core.utils import (
    ensure_store,
    ensure_string_type,
    verify_metadata_version,
)
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.utils import align_categories
from kartothek.serialization import (
    DataFrameSerializer,
    PredicatesType,
    default_serializer,
    filter_df_from_predicates,
)

LOGGER = logging.getLogger(__name__)

SINGLE_TABLE = "table"

_Literal = namedtuple("_Literal", ["column", "op", "value"])
_SplitPredicate = namedtuple("_SplitPredicate", ["key_part", "content_part"])

_METADATA_SCHEMA = {
    "partition_label": np.dtype("O"),
    "row_group_id": np.dtype(int),
    "row_group_compressed_size": np.dtype(int),
    "row_group_uncompressed_size": np.dtype(int),
    "number_rows_total": np.dtype(int),
    "number_row_groups": np.dtype(int),
    "serialized_size": np.dtype(int),
    "number_rows_per_row_group": np.dtype(int),
}

MetaPartitionInput = Optional[Union[pd.DataFrame, Sequence, "MetaPartition"]]


def _predicates_to_named(predicates):
    if predicates is None:
        return None
    return [[_Literal(*x) for x in conjunction] for conjunction in predicates]


def _combine_predicates(predicates, logical_conjunction):
    if not logical_conjunction:
        return predicates
    if predicates is None:
        return [logical_conjunction]
    combined_predicates = []
    for conjunction in predicates:
        new_conjunction = conjunction[:]
        for literal in logical_conjunction:
            new_conjunction.append(literal)
        combined_predicates.append(new_conjunction)
    return combined_predicates


def _initialize_store_for_metapartition(method, method_args, method_kwargs):

    for store_variable in ["store", "storage"]:
        if store_variable in method_kwargs:
            method_kwargs[store_variable] = ensure_store(method_kwargs[store_variable])
        else:
            method = cast(object, method)
            args = inspect.getfullargspec(method).args

            if store_variable in args:
                ix = args.index(store_variable)
                # reduce index since the argspec and method_args start counting differently due to self
                ix -= 1
                instantiated_store = ensure_store(method_args[ix])
                new_args = []
                for ix_method, arg in enumerate(method_args):
                    if ix_method != ix:
                        new_args.append(arg)
                    else:
                        new_args.append(instantiated_store)
                method_args = tuple(new_args)

    return method_args, method_kwargs


def _apply_to_list(method):
    """
    Decorate a MetaPartition method to act upon the internal list of metapartitions

    The methods must return a MetaPartition object!
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if not isinstance(self, MetaPartition):
            raise TypeError("Type unknown %s", type(self))

        result = self.as_sentinel()
        if len(self) == 0:
            raise RuntimeError("Invalid MetaPartition. No sub-partitions to act upon.")

        # Look whether there is a `store` in the arguments and instatiate it
        # this way we avoid multiple HTTP pools
        method_args, method_kwargs = _initialize_store_for_metapartition(
            method, method_args, method_kwargs
        )
        method_return = None  # declare for mypy
        if (len(self) == 1) and (self.label is None):
            result = method(self, *method_args, **method_kwargs)
        else:
            for mp in self:
                method_return = method(mp, *method_args, **method_kwargs)
                if not isinstance(method_return, MetaPartition):
                    raise ValueError(
                        "Method {} did not return a MetaPartition "
                        "but {}".format(method.__name__, type(method_return))
                    )
                if method_return.is_sentinel:
                    result = method_return
                else:
                    for mp in method_return:
                        result = result.add_metapartition(mp, schema_validation=False)
        if not isinstance(result, MetaPartition):
            raise ValueError(
                "Result for method {} is not a `MetaPartition` but {}".format(
                    method.__name__, type(method_return)
                )
            )
        return result

    return _impl


class MetaPartitionIterator(Iterator):
    def __init__(self, metapartition):
        self.metapartition = metapartition
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        current = self.metapartition
        if len(current) == 1:
            if current.label is None:
                raise StopIteration()

        if self.position >= len(current.metapartitions):
            raise StopIteration()
        else:
            mp_dict = current.metapartitions[self.position]
            # These are global attributes, i.e. the nested metapartitions do not carry these and need
            # to be added here
            mp_dict["metadata_version"] = current.metadata_version
            mp_dict["schema"] = current.schema
            mp_dict["partition_keys"] = current.partition_keys
            mp_dict["logical_conjunction"] = current.logical_conjunction
            mp_dict["table_name"] = current.table_name
            self.position += 1
            return MetaPartition.from_dict(mp_dict)

    next = __next__  # Python 2


class MetaPartition(Iterable):
    """
    Wrapper for kartothek partition which includes additional information
    about the parent dataset
    """

    def __init__(
        self,
        label: Optional[str],
        file: Optional[str] = None,
        table_name: str = SINGLE_TABLE,
        data: Optional[pd.DataFrame] = None,
        indices: Optional[Dict[Any, Any]] = None,
        metadata_version: Optional[int] = None,
        schema: Optional[SchemaWrapper] = None,
        partition_keys: Optional[Sequence[str]] = None,
        logical_conjunction: Optional[List[Tuple[Any, str, Any]]] = None,
    ):
        """
        Initialize the :mod:`kartothek.io` base class MetaPartition.

        The `MetaPartition` is used as a wrapper around the kartothek
        `Partition` and primarily deals with dataframe manipulations,
        in- and output to store.

        The :class:`kartothek.io_components.metapartition` is immutable, i.e. all member
        functions will return a new MetaPartition object where the new
        attribute is changed

        Parameters
        ----------
        label
            partition label
        files
            A dictionary with references to the files in store where the
            keys represent file labels and the keys file prefixes.
        metadata
            The metadata of the partition
        data
            A dictionary including the materialized in-memory DataFrames
            corresponding to the file references in `files`.
        indices
            Kartothek index dictionary,
        metadata_version
        table_meta
            The dataset table schemas
        partition_keys
            The dataset partition keys
        logical_conjunction
            A logical conjunction to assign to the MetaPartition. By assigning
            this, the MetaPartition will only be able to load data respecting
            this conjunction.
        """

        if metadata_version is None:
            self.metadata_version = naming.DEFAULT_METADATA_VERSION
        else:
            self.metadata_version = metadata_version
        verify_metadata_version(self.metadata_version)
        self.schema = schema
        self.table_name = table_name
        if data is not None and schema is None:
            self.schema = make_meta(
                data, origin=f"{table_name}/{label}", partition_keys=partition_keys
            )

        indices = indices or {}
        for column, index_dct in indices.items():
            if isinstance(index_dct, dict):
                indices[column] = ExplicitSecondaryIndex(
                    column=column, index_dct=index_dct
                )
        self.logical_conjunction = logical_conjunction
        self.metapartitions = [
            {
                "label": label,
                "data": data,
                "file": file or None,
                "indices": indices,
                "logical_conjunction": logical_conjunction,
            }
        ]
        self.partition_keys = partition_keys or []

    def __repr__(self):
        if len(self.metapartitions) > 1:
            label = "NESTED ({})".format(len(self.metapartitions))
        else:
            label = self.label
        return "<{_class} v{version} | {label} >".format(
            version=self.metadata_version, _class=self.__class__.__name__, label=label
        )

    def __len__(self):
        return len(self.metapartitions)

    def __iter__(self):
        return MetaPartitionIterator(self)

    def __getitem__(self, label):
        for mp in self:
            if mp.label == label:
                return mp
        raise KeyError("Metapartition doesn't contain partition `{}`".format(label))

    @property
    def data(self):
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `data` attribute is not allowed while nested"
            )
        assert isinstance(self.metapartitions[0], dict), self.metapartitions
        return self.metapartitions[0]["data"]

    @property
    def file(self) -> str:
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `files` attribute is not allowed while nested"
            )
        return cast(str, self.metapartitions[0]["file"])

    @property
    def is_sentinel(self) -> bool:
        return len(self.metapartitions) == 1 and self.label is None

    @property
    def label(self) -> str:
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `label` attribute is not allowed while nested"
            )
        assert isinstance(self.metapartitions[0], dict), self.metapartitions[0]
        return cast(str, self.metapartitions[0]["label"])

    @property
    def indices(self):
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `indices` attribute is not allowed while nested"
            )
        return self.metapartitions[0]["indices"]

    @property
    def partition(self) -> Partition:
        return Partition(label=self.label, files={self.table_name: self.file})

    def __eq__(self, other):
        if not isinstance(other, MetaPartition):
            return False

        if self.metadata_version != other.metadata_version:
            return False

        if self.schema is not None and not self.schema.equals(other.schema):
            return False

        if len(self.metapartitions) != len(other.metapartitions):
            return False

        # In the case both MetaPartitions are nested, we need to ensure a match
        # for all sub-partitions.
        # Since the label is unique, this can be used as a distinguishing key to sort and compare
        # the nested metapartitions.
        if len(self.metapartitions) > 1:
            for mp_self, mp_other in zip(
                sorted(self.metapartitions, key=lambda x: x["label"]),
                sorted(other.metapartitions, key=lambda x: x["label"]),
            ):
                if mp_self == mp_other:
                    continue
                # If a single metapartition does not match, the whole object is considered different
                return False
            return True

        # This is unnested only

        if self.label != other.label:
            return False

        if self.file != other.file:
            return False

        if self.data is not None and not self.data.equals(other.data):
            return False

        return True

    @staticmethod
    def from_partition(
        partition: Partition,
        data: Optional[pd.DataFrame] = None,
        indices: Optional[Dict] = None,
        metadata_version: Optional[int] = None,
        schema: Optional[SchemaWrapper] = None,
        partition_keys: Optional[List[str]] = None,
        logical_conjunction: Optional[List[Tuple[Any, str, Any]]] = None,
        table_name: str = SINGLE_TABLE,
    ):
        """
        Transform a kartothek :class:`~kartothek.core.partition.Partition` into a
        :class:`~kartothek.io_components.metapartition.MetaPartition`.

        Parameters
        ----------
        partition
            The kartothek partition to be wrapped
        data
            A dictionaries with materialised :class:`~pandas.DataFrame`
        indices : dict
            The index dictionary of the dataset
        schema
            Type metadata for each table, optional
        metadata_version
        partition_keys
            A list of the primary partition keys

        Returns
        -------
        :class:`~kartothek.io_components.metapartition.MetaPartition`
        """
        return MetaPartition(
            label=partition.label,
            file=partition.files[table_name],
            data=data,
            indices=indices,
            metadata_version=metadata_version,
            schema=schema,
            partition_keys=partition_keys,
            logical_conjunction=logical_conjunction,
            table_name=table_name,
        )

    def add_metapartition(
        self, metapartition: "MetaPartition", schema_validation: bool = True,
    ):
        """
        Adds a metapartition to the internal list structure to enable batch processing.

        Parameters
        ----------
        metapartition
            The MetaPartition to be added.
        schema_validation
            If True (default), ensure that the `table_meta` of both `MetaPartition` objects are the same
        """
        if self.is_sentinel:
            return metapartition

        existing_label = [mp_["label"] for mp_ in self.metapartitions]

        if any(
            [mp_["label"] in existing_label for mp_ in metapartition.metapartitions]
        ):
            raise RuntimeError(
                "Duplicate labels for nested metapartitions are not allowed!"
            )
        schema = metapartition.schema

        if schema_validation and schema:
            # This ensures that only schema-compatible metapartitions can be nested
            # The returned schema by validate_compatible is the reference schema with the most
            # information, i.e. the fewest null columns
            schema = validate_compatible([self.schema, metapartition.schema])

        new_object = MetaPartition(
            label="NestedMetaPartition",
            metadata_version=metapartition.metadata_version,
            schema=schema,
            partition_keys=metapartition.partition_keys or None,
            logical_conjunction=metapartition.logical_conjunction or None,
            table_name=metapartition.table_name,
        )

        # Add metapartition information to the new object
        new_metapartitions = self.metapartitions.copy()
        new_metapartitions.extend(metapartition.metapartitions.copy())
        new_object.metapartitions = new_metapartitions

        return new_object

    @staticmethod
    def from_dict(dct):
        """
        Create a :class:`~kartothek.io_components.metapartition.MetaPartition` from a dictionary.

        Parameters
        ----------
        dct : dict
            Dictionary containing constructor arguments as keys

        Returns
        -------

        """
        return MetaPartition(
            label=dct["label"],
            file=dct.get("file", None),
            data=dct.get("data", None),
            table_name=dct.get("table_name", SINGLE_TABLE),
            indices=dct.get("indices", {}),
            metadata_version=dct.get("metadata_version", None),
            schema=dct.get("schema", None),
            partition_keys=dct.get("partition_keys", None),
            logical_conjunction=dct.get("logical_conjunction", None),
        )

    def to_dict(self):
        return {
            "label": self.label,
            "file": self.file,
            "data": self.data,
            "indices": self.indices,
            "metadata_version": self.metadata_version,
            "schema": self.schema,
            "partition_keys": self.partition_keys,
            "logical_conjunction": self.logical_conjunction,
            "table_name": self.table_name,
        }

    @_apply_to_list
    def remove_dataframes(self):
        """
        Remove all dataframes from the metapartition in memory.
        """
        return self.copy(data=None)

    def _split_predicates_in_index_and_content(self, predicates):
        """
        Split a list of predicates in the parts that can be resolved by the
        partition columns and the ones that are persisted in the data file.
        """
        # Predicates are split in this function into the parts that apply to
        # the partition key columns `key_part` and the parts that apply to the
        # contents of the file `content_part`.
        split_predicates = []
        has_index_condition = False
        for conjunction in predicates:
            key_part = []
            content_part = []
            for literal in conjunction:
                if literal.column in self.partition_keys:
                    has_index_condition = True
                    key_part.append(literal)
                else:
                    content_part.append(literal)
            split_predicates.append(_SplitPredicate(key_part, content_part))
        return split_predicates, has_index_condition

    def _apply_partition_key_predicates(self, indices, split_predicates):
        """
        Apply the predicates to the partition_key columns and return the remaining
        predicates that should be pushed to the DataFrame serialiser.
        """
        # Construct a single line DF with the partition columns
        schema = self.schema
        index_df_dct = {}
        for column, value in indices:
            pa_dtype = schema[schema.get_field_index(column)].type
            value = IndexBase.normalize_value(pa_dtype, value)
            if pa.types.is_date(pa_dtype):
                index_df_dct[column] = pd.Series(
                    pd.to_datetime([value], infer_datetime_format=True)
                ).dt.date
            else:
                dtype = pa_dtype.to_pandas_dtype()
                index_df_dct[column] = pd.Series([value], dtype=dtype)
        index_df = pd.DataFrame(index_df_dct)

        filtered_predicates = []
        # We assume that indices on the partition level have been filtered out already in `dispatch_metapartitions`.
        # `filtered_predicates` should only contain predicates that can be evaluated on parquet level
        for conjunction in split_predicates:
            predicates = [conjunction.key_part]
            if (
                len(conjunction.key_part) == 0
                or len(
                    filter_df_from_predicates(
                        index_df, predicates, strict_date_types=True
                    )
                )
                > 0
            ):
                if len(conjunction.content_part) > 0:
                    filtered_predicates.append(conjunction.content_part)
                else:
                    # A condititon applies to the whole DataFrame, so we need to
                    # load all data.
                    return None
        return filtered_predicates

    @default_docs
    @_apply_to_list
    def load_dataframes(
        self,
        store: KeyValueStore,
        columns: Optional[Sequence[str]] = None,
        predicate_pushdown_to_io: bool = True,
        categoricals: Optional[Sequence[str]] = None,
        dates_as_object: bool = True,
        predicates: PredicatesType = None,
    ) -> "MetaPartition":
        """
        Load the dataframes of the partitions from store into memory.

        Parameters
        ----------
        tables
            If a list is supplied, only the given tables of the partition are
            loaded. If the given table does not exist it is ignored.

            Examples

            .. code::

                >>> part = MetaPartition(
                ...     label='part_label'
                ...     files={
                ...         'core': 'core_key_in_store',
                ...         'helper': 'helper_key_in_store'
                ...     }
                ...  )
                >>> part.data
                    {}
                >>> part = part.load_dataframes(store, ['core'])
                >>> part.data
                    {
                        'core': pd.DataFrame()
                    }

        """

        if categoricals is None:
            categoricals = []
        if not dates_as_object:
            warnings.warn(
                "The argument `date_as_object` is set to False. This argument will be deprecated and the future behaviour will be as if the paramere was set to `True`. Please migrate your code accordingly ahead of time.",
                DeprecationWarning,
            )

        LOGGER.debug("Loading internal dataframes of %s", self.label)
        if not self.file:
            # This used to raise, but the specs do not require this, so simply do a no op
            LOGGER.debug("Partition %s is empty and has no data.", self.label)
            return self
        predicates = _combine_predicates(predicates, self.logical_conjunction)
        predicates = _predicates_to_named(predicates)

        dataset_uuid, _, indices, _ = decode_key(self.file)

        # In case the columns only refer to the partition indices, we need to load at least a single column to
        # determine the length of the required dataframe.
        table_columns_to_io = columns

        filtered_predicates = predicates

        self = self.load_schema(dataset_uuid=dataset_uuid, store=store)

        # Filter predicates that would apply to this partition and remove the partition columns
        if predicates:
            # Check if there are predicates that match to the partition columns.
            # For these we need to check if the partition columns already falsify
            # the conditition.
            #
            # We separate these predicates into their index and their Parquet part.
            (
                split_predicates,
                has_index_condition,
            ) = self._split_predicates_in_index_and_content(predicates)

            filtered_predicates = []
            if has_index_condition:
                filtered_predicates = self._apply_partition_key_predicates(
                    indices, split_predicates
                )
            else:
                filtered_predicates = [pred.content_part for pred in split_predicates]

        # Remove partition_keys from table_columns_to_io
        if self.partition_keys and table_columns_to_io is not None:
            keys_to_remove = set(self.partition_keys) & set(table_columns_to_io)
            # This is done to not change the ordering of the list
            table_columns_to_io = [
                c for c in table_columns_to_io if c not in keys_to_remove
            ]

        start = time.time()
        df = DataFrameSerializer.restore_dataframe(
            key=self.file,
            store=store,
            columns=table_columns_to_io,
            categories=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            predicates=filtered_predicates,
            date_as_object=dates_as_object,
        )
        LOGGER.debug(
            "Loaded dataframe %s in %s seconds.", self.file, time.time() - start
        )
        # Metadata version >=4 parse the index columns and add them back to the dataframe

        df = self._reconstruct_index_columns(
            df=df,
            key_indices=indices,
            columns=columns,
            categories=categoricals,
            date_as_object=dates_as_object,
        )

        df.columns = df.columns.map(ensure_string_type)
        if columns is not None:
            # TODO: When the write-path ensures that all partitions have the same column set, this check can be
            #       moved before `DataFrameSerializer.restore_dataframe`. At the position of the current check we
            #       may want to double check the columns of the loaded DF and raise an exception indicating an
            #       inconsistent dataset state instead.
            missing_cols = set(columns).difference(df.columns)
            if missing_cols:
                raise ValueError(
                    "Columns cannot be found in stored dataframe: {}".format(
                        ", ".join(sorted(missing_cols))
                    )
                )

            if list(df.columns) != columns:
                df = df.reindex(columns=columns, copy=False)

        return self.copy(data=df)

    @_apply_to_list
    def load_schema(self, store: StoreInput, dataset_uuid: str) -> "MetaPartition":
        """
        Loads all table metadata in memory and stores it under the `tables` attribute

        """

        if self.schema is None:
            store = ensure_store(store)
            self.schema = read_schema_metadata(
                dataset_uuid=dataset_uuid, store=store, table=self.table_name
            )
        return self

    def _reconstruct_index_columns(
        self, df, key_indices, columns, categories, date_as_object
    ):
        if len(key_indices) == 0:
            return df

        original_columns = list(df.columns)
        zeros = np.zeros(len(df), dtype=int)
        schema = self.schema

        # One of the few places `inplace=True` makes a signifcant difference
        df.reset_index(drop=True, inplace=True)

        index_names = [primary_key for primary_key, _ in key_indices]
        # The index might already be part of the dataframe which is recovered from the parquet file.
        # In this case, still use the reconstructed index col to have consistent index columns behavior.
        # In this case the column in part of `original_columns` and must be removed to avoid duplication
        # in the column axis
        cleaned_original_columns = [
            orig for orig in original_columns if orig not in index_names
        ]
        if cleaned_original_columns != original_columns:
            # indexer call is slow, so only do that if really necessary
            df = df.reindex(columns=cleaned_original_columns, copy=False)

        for pos, (primary_key, value) in enumerate(key_indices):
            # If there are predicates, don't reconstruct the index if it wasn't requested
            if columns is not None and primary_key not in columns:
                continue

            pa_dtype = schema.field(primary_key).type
            dtype = pa_dtype.to_pandas_dtype()
            convert_to_date = False
            if date_as_object and pa_dtype in [pa.date32(), pa.date64()]:
                convert_to_date = True

            if isinstance(dtype, type):
                value = dtype(value)
            elif isinstance(dtype, np.dtype):
                if dtype == np.dtype("datetime64[ns]"):
                    value = pd.Timestamp(value)
                else:
                    value = dtype.type(value)
            else:
                raise RuntimeError(
                    "Unexepected object encountered: ({}, {})".format(
                        dtype, type(dtype)
                    )
                )
            if categories and primary_key in categories:
                if convert_to_date:
                    cats = pd.Series(value).dt.date
                else:
                    cats = [value]
                value = pd.Categorical.from_codes(zeros, categories=cats)
            else:
                if convert_to_date:
                    value = pd.Timestamp(value).to_pydatetime().date()
            df.insert(pos, primary_key, value)

        return df

    @_apply_to_list
    def validate_schema_compatible(
        self, store: StoreInput, dataset_uuid: str
    ) -> "MetaPartition":
        """
        Validates that the currently held DataFrames match the schema of the existing dataset.

        Parameters
        ----------
        store
            If it is a function, the result of calling it must be a KeyValueStore.
        dataset_uuid
            The dataset UUID the partition will be assigned to
        """

        # Load the reference meta of the existing dataset. Using the built-in
        # `load_all_table_meta` would not be helpful here as it would be a no-op
        # as we have already loaded the meta from the input DataFrame.
        store = ensure_store(store)
        reference_meta = read_schema_metadata(
            dataset_uuid=dataset_uuid, store=store, table=self.table_name
        )
        try:
            validate_compatible([self.schema, reference_meta])
        except ValueError as e:
            raise ValueError(
                f"Schemas for dataset '{dataset_uuid}' are not compatible!\n\n{e}"
            )
        return self

    @_apply_to_list
    def store_dataframes(
        self,
        store: StoreInput,
        dataset_uuid: str,
        df_serializer: Optional[DataFrameSerializer] = None,
    ) -> "MetaPartition":
        """
        Stores all dataframes of the MetaPartitions and registers the saved
        files under the `files` atrribute. The dataframe itself is deleted from memory.

        Parameters
        ----------
        store
            If it is a function, the result of calling it must be a KeyValueStore.
        dataset_uuid
            The dataset UUID the partition will be assigned to
        df_serializer
            Serialiser to be used to store the dataframe
        Returns
        -------
        MetaPartition
        """
        df_serializer = (
            df_serializer if df_serializer is not None else default_serializer()
        )

        key = get_partition_file_prefix(
            partition_label=self.label,
            dataset_uuid=dataset_uuid,
            metadata_version=self.metadata_version,
            table=self.table_name,
        )
        if self.data is not None:
            df = self.data
            try:
                file = df_serializer.store(store, key, df)
            except Exception as exc:
                try:
                    if isinstance(df, pd.DataFrame):
                        buf = io.StringIO()
                        df.info(buf=buf)
                        LOGGER.error(
                            "Writing dataframe failed.\n" "%s\n" "%s\n" "%s",
                            exc,
                            buf.getvalue(),
                            df.head(),
                        )
                    else:
                        LOGGER.error("Storage of dask dataframe failed.")
                        pass
                finally:
                    raise

            new_metapartition = self.copy(file=file, data=None)

            return new_metapartition
        else:
            return self

    @_apply_to_list
    def apply(self, func: Callable, type_safe: bool = False,) -> "MetaPartition":
        """
        Applies a given function to all dataframes of the MetaPartition.

        Parameters
        ----------
        func
            A callable accepting and returning a :class:`pandas.DataFrame`
        uuid :
            The changed dataset is assigned a new UUID.
        type_safe
            If the transformation is type-safe, optimizations can be applied

        """

        new_data = func(self.data)

        if type_safe:
            new_schema = self.schema
        else:
            new_schema = make_meta(
                new_data, origin=self.label, partition_keys=self.partition_keys,
            )
        return self.copy(data=new_data, schema=new_schema)

    def as_sentinel(self):
        """"""
        return MetaPartition(
            None,
            metadata_version=self.metadata_version,
            partition_keys=self.partition_keys,
        )

    def copy(self, **kwargs):
        """
        Creates a shallow copy where the kwargs overwrite existing attributes
        """

        def _renormalize_meta(meta):
            if "partition_keys" in kwargs and meta is not None:
                pk = kwargs["partition_keys"]
                return normalize_column_order(meta, pk)
            else:
                return meta

        metapartitions = kwargs.get("metapartitions", None) or []
        metapartitions.extend(self.metapartitions)
        if len(metapartitions) > 1:
            first_mp = metapartitions.pop()
            mp_parent = MetaPartition(
                label=first_mp.get("label"),
                file=first_mp.get("file"),
                data=first_mp.get("data"),
                indices=first_mp.get("indices"),
                metadata_version=self.metadata_version,
                schema=_renormalize_meta(kwargs.get("schema", self.schema)),
                partition_keys=kwargs.get("partition_keys", self.partition_keys),
                logical_conjunction=kwargs.get(
                    "logical_conjunction", self.logical_conjunction
                ),
                table_name=kwargs.get("table_name", self.table_name),
            )
            for mp in metapartitions:
                mp_parent = mp_parent.add_metapartition(
                    MetaPartition(
                        label=mp.get("label"),
                        file=mp.get("file"),
                        data=mp.get("data"),
                        indices=mp.get("indices"),
                        metadata_version=self.metadata_version,
                        schema=_renormalize_meta(kwargs.get("schema", self.schema)),
                        partition_keys=kwargs.get(
                            "partition_keys", self.partition_keys
                        ),
                        logical_conjunction=kwargs.get(
                            "logical_conjunction", self.logical_conjunction
                        ),
                        table_name=kwargs.get("table_name", self.table_name),
                    ),
                    schema_validation=False,
                )
            return mp_parent
        else:
            mp = MetaPartition(
                label=kwargs.get("label", self.label),
                file=kwargs.get("file", self.file),
                data=kwargs.get("data", self.data),
                indices=kwargs.get("indices", self.indices),
                metadata_version=kwargs.get("metadata_version", self.metadata_version),
                schema=_renormalize_meta(kwargs.get("schema", self.schema)),
                partition_keys=kwargs.get("partition_keys", self.partition_keys),
                logical_conjunction=kwargs.get(
                    "logical_conjunction", self.logical_conjunction
                ),
                table_name=kwargs.get("table_name", self.table_name),
            )
            return mp

    @_apply_to_list
    def build_indices(self, columns: Iterable[str]):
        """
        This builds the indices for this metapartition for the given columns. The indices for the passed columns
        are rebuilt, so exisiting index entries in the metapartition are overwritten.

        :param columns: A list of columns from which the indices over all dataframes in the metapartition
            are overwritten
        :return: self
        """
        if self.label is None:
            return self

        new_indices = {}
        for col in columns:

            possible_values: Set[str] = set()

            df = self.data
            if not self.is_sentinel and col not in df:
                raise RuntimeError(
                    "Column `{corrupt_col}` could not be found in the partition `{partition_label}` Please check for any typos and validate your dataset.".format(
                        corrupt_col=col, partition_label=self.label
                    )
                )

            possible_values = possible_values | set(df[col].dropna().unique())

            if self.schema is not None:
                dtype = self.schema.field(col).type
            else:
                dtype = None

            new_index = ExplicitSecondaryIndex(
                column=col,
                index_dct={value: [self.label] for value in possible_values},
                dtype=dtype,
            )
            if (col in self.indices) and self.indices[col].loaded:
                new_indices[col] = self.indices[col].update(new_index)
            else:
                new_indices[col] = new_index

        return self.copy(indices=new_indices)

    @_apply_to_list
    def partition_on(self, partition_on: Union[str, Sequence[str]]):
        """
        Partition all dataframes assigned to this MetaPartition according the the given columns.

        If the MetaPartition object contains index information, the information is split in such a way that they
        reference the new partitions.

        In case a requested partition column is not existent in **all** tables, a KeyError is raised.

        All output partitions are re-assigned labels encoding the partitioned columns (urlencoded)

        Examples::

            >>> import pandas as pd
            >>> from kartothek.io_components.metapartition import MetaPartition
            >>> mp = MetaPartition(
            ...     label='partition_label',
            ...     data={
            ...         "Table1": pd.DataFrame({
            ...             'P': [1, 2, 1, 2],
            ...             'L': [1, 1, 2, 2]
            ...         })
            ...     }
            ... )
            >>> repartitioned_mp = mp.partition_on(['P', 'L'])
            >>> assert [mp["label"] for mp in repartitioned_mp.metapartitions] == [
            ...     "P=1/L=1/partition_label",
            ...     "P=1/L=2/partition_label",
            ...     "P=2/L=1/partition_label",
            ...     "P=2/L=2/partition_label"
            ... ]

        Parameters
        ----------
        partition_on


        """
        if partition_on == self.partition_keys:
            return self

        for partition_column in partition_on:
            if partition_column in self.indices:
                raise ValueError(
                    "Trying to `partition_on` on a column with an explicit index!"
                )
        if self.is_sentinel:
            return self.copy(partition_keys=partition_on)
        else:
            new_mp = self.as_sentinel().copy(
                partition_keys=partition_on,
                schema=normalize_column_order(self.schema, partition_on),
            )

        if isinstance(partition_on, str):
            partition_on = [partition_on]
        partition_on = self._ensure_compatible_partitioning(partition_on)

        new_data = self._partition_data(partition_on)

        for label, data in new_data.items():
            tmp_mp = MetaPartition(
                label=label,
                file=self.file,
                data=data,
                metadata_version=self.metadata_version,
                indices={},
                schema=normalize_column_order(self.schema, partition_on).with_origin(
                    f"{label}"
                ),
                partition_keys=partition_on,
                table_name=self.table_name,
            )
            new_mp = new_mp.add_metapartition(tmp_mp, schema_validation=False)
        if self.indices:
            new_mp = new_mp.build_indices(columns=self.indices.keys())
        return new_mp

    def _ensure_compatible_partitioning(self, partition_on):
        if (
            not self.partition_keys
            or self.partition_keys
            and (len(partition_on) >= len(self.partition_keys))
            and (self.partition_keys == partition_on[: len(self.partition_keys)])
        ):
            return partition_on[len(self.partition_keys) :]
        else:
            raise ValueError(
                "Incompatible partitioning encountered. `partition_on` needs to include the already "
                "existing partition keys and must preserve their order.\n"
                "Current partition keys: `{}`\n"
                "Partition on called with: `{}`".format(
                    self.partition_keys, partition_on
                )
            )

    def _partition_data(self, partition_on):
        existing_indices, base_label = decode_key("uuid/table/{}".format(self.label))[
            2:
        ]
        dct = dict()
        df = self.data

        # Check that data sizes do not change. This might happen if the
        # groupby below drops data, e.g. nulls
        size_after = 0
        size_before = len(df)

        # Implementation from pyarrow
        # See https://github.com/apache/arrow/blob/b33dfd9c6bd800308bb1619b237dbf24dea159be/python/pyarrow/parquet.py#L1030  # noqa: E501

        # column sanity checks
        data_cols = set(df.columns).difference(partition_on)
        missing_po_cols = set(partition_on).difference(df.columns)
        if missing_po_cols:
            raise ValueError(
                "Partition column(s) missing: {}".format(
                    ", ".join(sorted(missing_po_cols))
                )
            )
        if len(data_cols) == 0:
            raise ValueError("No data left to save outside partition columns")

        # To be aligned with open source tooling we drop the index columns and recreate
        # them upon reading as it is done by fastparquet and pyarrow
        partition_keys = [df[col] for col in partition_on]

        # # The handling of empty dfs is not part of the arrow implementation
        # if df.empty:
        #     return {}

        data_df = df.drop(partition_on, axis="columns")
        for value, group in data_df.groupby(by=partition_keys, sort=False):
            partitioning_info = []
            if pd.api.types.is_scalar(value):
                value = [value]
            if existing_indices:
                partitioning_info.extend(quote_indices(existing_indices))
            partitioning_info.extend(quote_indices(zip(partition_on, value)))
            partitioning_info.append(base_label)
            new_label = "/".join(partitioning_info)

            if new_label not in dct:
                dct[new_label] = {}
            dct[new_label] = group
            size_after += len(group)

        if size_before != size_after:
            raise ValueError(
                f"Original dataframe size ({size_before} rows) does not "
                f"match new dataframe size ({size_after} rows). "
                f"Hint: you may see this if you are trying to use `partition_on` on a column with null values."
            )

        return dct

    @staticmethod
    def merge_indices(metapartitions):
        list_of_indices = []
        for mp in metapartitions:
            for sub_mp in mp:
                if sub_mp.indices:
                    list_of_indices.append(sub_mp.indices)
        return merge_indices_algo(list_of_indices)

    @staticmethod
    def _merge_labels(metapartitions, label_merger=None):
        # Use the shortest of available labels since this has to be the partition
        #  label prefix
        new_label = None
        # FIXME: This is probably not compatible with >= v3
        if label_merger is None:
            for mp in metapartitions:
                label = mp.label
                if new_label is None or len(label) < len(new_label):
                    new_label = label
                    continue
        else:
            new_label = label_merger([mp.label for mp in metapartitions])

        return new_label

    @staticmethod
    def concat_metapartitions(metapartitions, label_merger=None):
        LOGGER.debug("Concatenating metapartitions")

        new_metadata_version = -1
        data = []
        schema = []
        for mp in metapartitions:
            new_metadata_version = max(new_metadata_version, mp.metadata_version)
            data.append(mp.data)
            schema.append(mp.schema)
            # Don't care about the partition_keys. If we try to merge
            # MetaPartitions without alignment the schemas won't match.
            partition_keys = mp.partition_keys

        categoricals = [
            col
            for col, dtype in data[0].items()
            if pd.api.types.is_categorical_dtype(dtype)
        ]
        if categoricals:
            data = align_categories(data, categoricals)
        new_df = pd.concat(data)

        new_schema = validate_compatible(schema)

        new_label = MetaPartition._merge_labels(metapartitions, label_merger)

        new_mp = MetaPartition(
            label=new_label,
            data=new_df,
            metadata_version=new_metadata_version,
            schema=new_schema,
            partition_keys=partition_keys,
        )

        return new_mp

    @_apply_to_list
    def delete_from_store(
        self, dataset_uuid: Any, store: StoreInput
    ) -> "MetaPartition":
        store = ensure_store(store)
        # Delete data first
        store.delete(self.file)
        return self.copy(file=None, data=None)

    def get_parquet_metadata(self, store: StoreInput) -> pd.DataFrame:
        """
        Retrieve the parquet metadata for the MetaPartition.
        Especially relevant for calculating dataset statistics.

        Parameters
        ----------
        store
          A factory function providing a KeyValueStore
        table_name
          Name of the kartothek table for which the statistics should be retrieved

        Returns
        -------
        pd.DataFrame
          A DataFrame with relevant parquet metadata
        """
        store = ensure_store(store)

        data = {}
        with store.open(self.file) as fd:  # type: ignore
            pq_metadata = pa.parquet.ParquetFile(fd).metadata

        data = {
            "partition_label": self.label,
            "serialized_size": pq_metadata.serialized_size,
            "number_rows_total": pq_metadata.num_rows,
            "number_row_groups": pq_metadata.num_row_groups,
            "row_group_id": [],
            "number_rows_per_row_group": [],
            "row_group_compressed_size": [],
            "row_group_uncompressed_size": [],
        }
        for rg_ix in range(pq_metadata.num_row_groups):
            rg = pq_metadata.row_group(rg_ix)
            data["row_group_id"].append(rg_ix)
            data["number_rows_per_row_group"].append(rg.num_rows)
            data["row_group_compressed_size"].append(rg.total_byte_size)
            data["row_group_uncompressed_size"].append(
                sum(
                    rg.column(col_ix).total_uncompressed_size
                    for col_ix in range(rg.num_columns)
                )
            )

        df = pd.DataFrame(data=data, columns=_METADATA_SCHEMA.keys())
        df = df.astype(_METADATA_SCHEMA)
        return df


def _unique_label(label_list):
    label = os.path.commonprefix(label_list)
    if len(label) == 0:
        label = "_".join(label_list)
    while len(label) > 0 and not label[-1].isalnum():
        label = label[:-1]
    return label


def partition_labels_from_mps(mps: List[MetaPartition]) -> List[str]:
    """
    Get a list of partition labels, flattening any nested meta partitions in the input and ignoring sentinels.

    """
    partition_labels = []
    for mp in mps:
        if len(mp) > 1:
            for nested_mp in mp:
                if not nested_mp.is_sentinel:
                    partition_labels.append(nested_mp.label)
        else:
            if not mp.is_sentinel:
                partition_labels.append(mp.label)
    return partition_labels


def parse_input_to_metapartition(
    obj: MetaPartitionInput,
    table_name: str = SINGLE_TABLE,
    metadata_version: Optional[int] = None,
) -> MetaPartition:
    """
    Parses given user input and return a MetaPartition

    The expected input is a :class:`pandas.DataFrame` or a list of
    :class:`pandas.DataFrame`.

    Every element of the list will be treated as a dedicated user input and will
    result in a physical file, if not specified otherwise.

    Parameters
    ----------
    obj
    table_name
        The table name assigned to the partitions
    metadata_version
        The kartothek dataset specification version
    """

    if obj is None:
        obj = []
    if isinstance(obj, list):
        if len(obj) == 0:
            return MetaPartition(label=None, metadata_version=metadata_version)
        first_element = obj[0]
        mp = parse_input_to_metapartition(
            obj=first_element, metadata_version=metadata_version, table_name=table_name,
        )
        for mp_in in obj[1:]:
            mp = mp.add_metapartition(
                parse_input_to_metapartition(
                    obj=mp_in, metadata_version=metadata_version, table_name=table_name,
                )
            )
    elif isinstance(obj, pd.DataFrame):
        mp = MetaPartition(
            label=gen_uuid(),
            data=obj,
            metadata_version=metadata_version,
            table_name=table_name,
        )
    elif isinstance(obj, MetaPartition):
        return obj
    else:
        raise ValueError(
            f"Unexpected type during parsing encountered: ({type(obj)}, {obj})"
        )

    return mp
