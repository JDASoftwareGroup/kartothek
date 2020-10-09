import inspect
import io
import logging
import os
import time
import warnings
from collections import defaultdict, namedtuple
from copy import copy
from functools import wraps
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Set, Union, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from kartothek.core import naming
from kartothek.core.common_metadata import (
    make_meta,
    normalize_column_order,
    read_schema_metadata,
    validate_compatible,
    validate_shared_columns,
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
from kartothek.io_components.utils import _ensure_valid_indices, combine_metadata
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

_MULTI_TABLE_DICT_LIST = Dict[str, Iterable[str]]


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
            mp_dict["dataset_metadata"] = current.dataset_metadata
            mp_dict["metadata_version"] = current.metadata_version
            mp_dict["table_meta"] = current.table_meta
            mp_dict["partition_keys"] = current.partition_keys
            mp_dict["logical_conjunction"] = current.logical_conjunction
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
        label,
        files=None,
        metadata=None,
        data=None,
        dataset_metadata=None,
        indices: Optional[Dict[Any, Any]] = None,
        metadata_version=None,
        table_meta=None,
        partition_keys=None,
        logical_conjunction=None,
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
        label : basestring
            partition label
        files : dict, optional
            A dictionary with references to the files in store where the
            keys represent file labels and the keys file prefixes.
        metadata : dict, optional
            The metadata of the partition
        data : dict, optional
            A dictionary including the materialized in-memory DataFrames
            corresponding to the file references in `files`.
        dataset_metadata : dict, optional
            The metadata of the original dataset
        indices : dict, optional
            Kartothek index dictionary,
        metadata_version : int, optional
        table_meta: Dict[str, SchemaWrapper]
            The dataset table schemas
        partition_keys: List[str]
            The dataset partition keys
        logical_conjunction: List[Tuple[object, str, object]]
            A logical conjunction to assign to the MetaPartition. By assigning
            this, the MetaPartition will only be able to load data respecting
            this conjunction.
        """

        if metadata_version is None:
            self.metadata_version = naming.DEFAULT_METADATA_VERSION
        else:
            self.metadata_version = metadata_version
        verify_metadata_version(self.metadata_version)
        self.table_meta = table_meta if table_meta else {}
        if isinstance(data, dict) and (len(self.table_meta) == 0):
            for table, df in data.items():
                if df is not None:
                    self.table_meta[table] = make_meta(
                        df,
                        origin="{}/{}".format(table, label),
                        partition_keys=partition_keys,
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
                "data": data or {},
                "files": files or {},
                "indices": indices,
                "logical_conjunction": logical_conjunction,
            }
        ]
        self.dataset_metadata = dataset_metadata or {}
        self.partition_keys = partition_keys or []

    def __repr__(self):
        if len(self.metapartitions) > 1:
            label = "NESTED ({})".format(len(self.metapartitions))
        else:
            label = self.label
        return "<{_class} v{version} | {label} | tables {tables} >".format(
            version=self.metadata_version,
            _class=self.__class__.__name__,
            label=label,
            tables=sorted(set(self.table_meta.keys())),
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
    def files(self):
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `files` attribute is not allowed while nested"
            )
        return self.metapartitions[0]["files"]

    @property
    def is_sentinel(self):
        return len(self.metapartitions) == 1 and self.label is None

    @property
    def label(self):
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `label` attribute is not allowed while nested"
            )
        assert isinstance(self.metapartitions[0], dict), self.metapartitions[0]
        return self.metapartitions[0]["label"]

    @property
    def indices(self):
        if len(self.metapartitions) > 1:
            raise AttributeError(
                "Accessing `indices` attribute is not allowed while nested"
            )
        return self.metapartitions[0]["indices"]

    @property
    def tables(self):
        return list(set(self.data.keys()).union(set(self.files.keys())))

    @property
    def partition(self):
        return Partition(label=self.label, files=self.files)

    def __eq__(self, other):
        if not isinstance(other, MetaPartition):
            return False

        if self.metadata_version != other.metadata_version:
            return False

        for table, meta in self.table_meta.items():
            # https://issues.apache.org/jira/browse/ARROW-5873
            other_meta = other.table_meta.get(table, None)
            if other_meta is None:
                return False
            if not meta.equals(other_meta):
                return False

        if self.dataset_metadata != other.dataset_metadata:
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

        self_keys = set(self.data.keys())
        other_keys = set(other.data.keys())
        if not (self_keys == other_keys):
            return False

        if self.label != other.label:
            return False

        if self.files != other.files:
            return False

        for label, df in self.data.items():
            if not (df.equals(other.data[label])):
                return False

        return True

    @staticmethod
    def from_partition(
        partition,
        data=None,
        dataset_metadata=None,
        indices=None,
        metadata_version=None,
        table_meta=None,
        partition_keys=None,
        logical_conjunction=None,
    ):
        """
        Transform a kartothek :class:`~kartothek.core.partition.Partition` into a
        :class:`~kartothek.io_components.metapartition.MetaPartition`.

        Parameters
        ----------
        partition : :class:`~kartothek.core.partition.Partition`
            The kartothek partition to be wrapped
        data : dict, optional
            A dictionaries with materialised :class:`~pandas.DataFrame`
        dataset_metadata : dict of basestring, optional
            The metadata of the original dataset
        indices : dict
            The index dictionary of the dataset
        table_meta: Union[None, Dict[String, pyarrow.Schema]]
            Type metadata for each table, optional
        metadata_version: int, optional
        partition_keys: Union[None, List[String]]
            A list of the primary partition keys
        Returns
        -------
        :class:`~kartothek.io_components.metapartition.MetaPartition`
        """
        return MetaPartition(
            label=partition.label,
            files=partition.files,
            data=data,
            dataset_metadata=dataset_metadata,
            indices=indices,
            metadata_version=metadata_version,
            table_meta=table_meta,
            partition_keys=partition_keys,
            logical_conjunction=logical_conjunction,
        )

    def add_metapartition(
        self, metapartition, metadata_merger=None, schema_validation=True
    ):
        """
        Adds a metapartition to the internal list structure to enable batch processing.

        The top level `dataset_metadata` dictionary is combined with the existing dict and
        all other attributes are stored in the `metapartitions` list

        Parameters
        ----------
        metapartition: [MetaPartition]
            The MetaPartition to be added.
        metadata_merger: [callable]
            A callable to perform the metadata merge. By default [kartothek.io_components.utils.combine_metadata] is used
        schema_validation : [bool]
            If True (default), ensure that the `table_meta` of both `MetaPartition` objects are the same
        """
        if self.is_sentinel:
            return metapartition

        table_meta = metapartition.table_meta
        existing_label = [mp_["label"] for mp_ in self.metapartitions]

        if any(
            [mp_["label"] in existing_label for mp_ in metapartition.metapartitions]
        ):
            raise RuntimeError(
                "Duplicate labels for nested metapartitions are not allowed!"
            )

        if schema_validation:
            table_meta = {}
            for table, meta in self.table_meta.items():
                other = metapartition.table_meta.get(table, None)
                # This ensures that only schema-compatible metapartitions can be nested
                # The returned schema by validate_compatible is the reference schema with the most
                # information, i.e. the fewest null columns
                table_meta[table] = validate_compatible([meta, other])

        metadata_merger = metadata_merger or combine_metadata
        new_dataset_metadata = metadata_merger(
            [self.dataset_metadata, metapartition.dataset_metadata]
        )

        new_object = MetaPartition(
            label="NestedMetaPartition",
            dataset_metadata=new_dataset_metadata,
            metadata_version=metapartition.metadata_version,
            table_meta=table_meta,
            partition_keys=metapartition.partition_keys or None,
            logical_conjunction=metapartition.logical_conjunction or None,
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
            files=dct.get("files", {}),
            metadata=dct.get("metadata", {}),
            data=dct.get("data", {}),
            indices=dct.get("indices", {}),
            metadata_version=dct.get("metadata_version", None),
            dataset_metadata=dct.get("dataset_metadata", {}),
            table_meta=dct.get("table_meta", {}),
            partition_keys=dct.get("partition_keys", None),
            logical_conjunction=dct.get("logical_conjunction", None),
        )

    def to_dict(self):
        return {
            "label": self.label,
            "files": self.files or {},
            "data": self.data or {},
            "indices": self.indices,
            "metadata_version": self.metadata_version,
            "dataset_metadata": self.dataset_metadata,
            "table_meta": self.table_meta,
            "partition_keys": self.partition_keys,
            "logical_conjunction": self.logical_conjunction,
        }

    @_apply_to_list
    def remove_dataframes(self):
        """
        Remove all dataframes from the metapartition in memory.
        """
        return self.copy(data={})

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

    def _apply_partition_key_predicates(self, table, indices, split_predicates):
        """
        Apply the predicates to the partition_key columns and return the remaining
        predicates that should be pushed to the DataFrame serialiser.
        """
        # Construct a single line DF with the partition columns
        schema = self.table_meta[table]
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
        store: StoreInput,
        tables: _MULTI_TABLE_DICT_LIST = None,
        columns: _MULTI_TABLE_DICT_LIST = None,
        predicate_pushdown_to_io: bool = True,
        categoricals: _MULTI_TABLE_DICT_LIST = None,
        dates_as_object: bool = False,
        predicates: PredicatesType = None,
    ) -> "MetaPartition":
        """
        Load the dataframes of the partitions from store into memory.

        Parameters
        ----------
        tables : list of string, optional
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
        if columns is None:
            columns = {}
        elif set(columns).difference(self.tables):
            raise (
                ValueError(
                    "You are trying to read columns from invalid table(s): {}".format(
                        set(columns).difference(self.tables)
                    )
                )
            )

        if categoricals is None:
            categoricals = {}

        LOGGER.debug("Loading internal dataframes of %s", self.label)
        if len(self.files) == 0:
            # This used to raise, but the specs do not require this, so simply do a no op
            LOGGER.debug("Partition %s is empty and has not tables/files", self.label)
            return self
        new_data = copy(self.data)
        predicates = _combine_predicates(predicates, self.logical_conjunction)
        predicates = _predicates_to_named(predicates)

        for table, key in self.files.items():
            table_columns = columns.get(table, None)
            categories = categoricals.get(table, None)
            dataset_uuid, _, indices, file_name = decode_key(key)
            if tables and table not in tables:
                continue

            # In case the columns only refer to the partition indices, we need to load at least a single column to
            # determine the length of the required dataframe.
            if table_columns is None:
                table_columns_to_io = None
            else:
                table_columns_to_io = table_columns

            filtered_predicates = predicates

            self._load_table_meta(dataset_uuid=dataset_uuid, table=table, store=store)

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
                        table, indices, split_predicates
                    )
                else:
                    filtered_predicates = [
                        pred.content_part for pred in split_predicates
                    ]

            # Remove partition_keys from table_columns_to_io
            if self.partition_keys and table_columns_to_io is not None:
                keys_to_remove = set(self.partition_keys) & set(table_columns_to_io)
                # This is done to not change the ordering of the list
                table_columns_to_io = [
                    c for c in table_columns_to_io if c not in keys_to_remove
                ]

            start = time.time()
            df = DataFrameSerializer.restore_dataframe(
                key=key,
                store=store,
                columns=table_columns_to_io,
                categories=categories,
                predicate_pushdown_to_io=predicate_pushdown_to_io,
                predicates=filtered_predicates,
                date_as_object=dates_as_object,
            )
            LOGGER.debug("Loaded dataframe %s in %s seconds.", key, time.time() - start)
            # Metadata version >=4 parse the index columns and add them back to the dataframe

            df = self._reconstruct_index_columns(
                df=df,
                key_indices=indices,
                table=table,
                columns=table_columns,
                categories=categories,
                date_as_object=dates_as_object,
            )

            df.columns = df.columns.map(ensure_string_type)
            if table_columns is not None:
                # TODO: When the write-path ensures that all partitions have the same column set, this check can be
                #       moved before `DataFrameSerializer.restore_dataframe`. At the position of the current check we
                #       may want to double check the columns of the loaded DF and raise an exception indicating an
                #       inconsistent dataset state instead.
                missing_cols = set(table_columns).difference(df.columns)
                if missing_cols:
                    raise ValueError(
                        "Columns cannot be found in stored dataframe: {}".format(
                            ", ".join(sorted(missing_cols))
                        )
                    )

                if list(df.columns) != table_columns:
                    df = df.reindex(columns=table_columns, copy=False)
            new_data[table] = df
        return self.copy(data=new_data)

    @_apply_to_list
    def load_all_table_meta(
        self, store: StoreInput, dataset_uuid: str
    ) -> "MetaPartition":
        """
        Loads all table metadata in memory and stores it under the `tables` attribute

        """
        for table in self.files:
            self._load_table_meta(dataset_uuid, table, store)
        return self

    def _load_table_meta(
        self, dataset_uuid: str, table: str, store: StoreInput
    ) -> "MetaPartition":
        if table not in self.table_meta:
            _common_metadata = read_schema_metadata(
                dataset_uuid=dataset_uuid, store=store, table=table
            )
            self.table_meta[table] = _common_metadata
        return self

    def _reconstruct_index_columns(
        self, df, key_indices, table, columns, categories, date_as_object
    ):
        if len(key_indices) == 0:
            return df

        original_columns = list(df.columns)
        zeros = np.zeros(len(df), dtype=int)
        schema = self.table_meta[table]

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
    def merge_dataframes(
        self, left, right, output_label, merge_func=pd.merge, merge_kwargs=None
    ):
        """
        Merge internal dataframes.

        The two referenced dataframes are removed from the internal list and
        the newly created dataframe is added.

        The merge itself can be completely customized by supplying a
        callable `merge_func(left_df, right_df, **merge_kwargs)` which can
        handle data pre-processing as well as the merge itself.

        The files attribute of the result will be empty since the in-memory
        DataFrames are no longer representations of the referenced files.

        Parameters
        ----------
        left : basestring
            Category of the left dataframe.
        right : basestring
            Category of the right dataframe.
        output_label : basestring
            Category for the newly created dataframe
        merge_func : callable, optional
            The function to take care of the merge. By default: pandas.merge.
            The function should have the signature
            :func:`func(left_df, right_df, **kwargs)`
        merge_kwargs : dict
            Keyword arguments which should be supplied to the merge function

        Returns
        -------
        MetaPartition

        """
        # Shallow copy
        new_data = copy(self.data)
        if merge_kwargs is None:
            merge_kwargs = {}

        left_df = new_data.pop(left)
        right_df = new_data.pop(right)

        LOGGER.debug("Merging internal dataframes of %s", self.label)

        try:
            df_merged = merge_func(left_df, right_df, **merge_kwargs)
        except TypeError:
            LOGGER.error(
                "Tried to merge using %s with\n left:%s\nright:%s\n " "kwargs:%s",
                merge_func.__name__,
                left_df.head(),
                right_df.head(),
                merge_kwargs,
            )
            raise

        new_data[output_label] = df_merged
        new_table_meta = copy(self.table_meta)
        # The tables are no longer part of the MetaPartition, thus also drop
        # their schema.
        del new_table_meta[left]
        del new_table_meta[right]
        new_table_meta[output_label] = make_meta(
            df_merged,
            origin="{}/{}".format(output_label, self.label),
            partition_keys=self.partition_keys,
        )
        return self.copy(files={}, data=new_data, table_meta=new_table_meta)

    @_apply_to_list
    def validate_schema_compatible(
        self, store: StoreInput, dataset_uuid: str
    ) -> "MetaPartition":
        """
        Validates that the currently held DataFrames match the schema of the existing dataset.

        Parameters
        ----------
        store: KeyValueStore or callable
            If it is a function, the result of calling it must be a KeyValueStore.
        dataset_uuid: str
            The dataset UUID the partition will be assigned to
        """

        # Load the reference meta of the existing dataset. Using the built-in
        # `load_all_table_meta` would not be helpful here as it would be a no-op
        # as we have already loaded the meta from the input DataFrame.
        reference_meta = {}
        for table in self.table_meta:
            _common_metadata = read_schema_metadata(
                dataset_uuid=dataset_uuid, store=store, table=table
            )
            reference_meta[table] = _common_metadata

        result = {}
        for table, schema in self.table_meta.items():
            try:
                result[table] = validate_compatible([schema, reference_meta[table]])
            except ValueError as e:
                raise ValueError(
                    "Schemas for table '{table}' of dataset '{dataset_uuid}' are not compatible!\n\n{e}".format(
                        table=table, dataset_uuid=dataset_uuid, e=e
                    )
                )
        validate_shared_columns(list(result.values()))

        return self

    @_apply_to_list
    def store_dataframes(
        self,
        store: StoreInput,
        dataset_uuid: str,
        df_serializer: Optional[DataFrameSerializer] = None,
        store_metadata: bool = False,
        metadata_storage_format: Optional[str] = None,
    ) -> "MetaPartition":
        """
        Stores all dataframes of the MetaPartitions and registers the saved
        files under the `files` atrribute. The dataframe itself is deleted from memory.

        Parameters
        ----------
        store: KeyValueStore or callable
            If it is a function, the result of calling it must be a KeyValueStore.
        dataset_uuid: str
            The dataset UUID the partition will be assigned to
        df_serializer : kartothek.serialization.DataFrameSerializer
            Serialiser to be used to store the dataframe
        Returns
        -------
        MetaPartition
        """
        df_serializer = (
            df_serializer if df_serializer is not None else default_serializer()
        )
        file_dct = {}

        for table, df in self.data.items():
            key = get_partition_file_prefix(
                partition_label=self.label,
                dataset_uuid=dataset_uuid,
                table=table,
                metadata_version=self.metadata_version,
            )
            LOGGER.debug("Store dataframe for table `%s` to %s ...", table, key)
            try:
                file_dct[table] = df_serializer.store(store, key, df)
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
            LOGGER.debug("Storage of dataframe for table `%s` successful", table)

        new_metapartition = self.copy(files=file_dct, data={})

        return new_metapartition

    @_apply_to_list
    def concat_dataframes(self):
        """
        Concatenates all dataframes with identical columns.

        In case of changes on the dataframes, the files attribute will be
        emptied since the in-memory DataFrames are no longer representations
        of the referenced files.

        Returns
        -------
        MetaPartition
            A metapartition where common column dataframes are merged. The
            file attribute will be empty since there is no direct relation
            between the referenced files and the in-memory dataframes anymore

        """

        count_cols = defaultdict(list)
        for label, df in self.data.items():
            # List itself is not hashable
            key = "".join(sorted(df.columns))
            count_cols[key].append((label, df))
        is_modified = False
        new_data = {}
        for _, tuple_list in count_cols.items():
            if len(tuple_list) > 1:
                is_modified = True
                data = [x[1] for x in tuple_list]
                label = _unique_label([x[0] for x in tuple_list])
                new_data[label] = pd.concat(data).reset_index(drop=True)
            else:
                label, df = tuple_list[0]
                new_data[label] = df
        new_table_meta = {
            label: make_meta(
                df,
                origin="{}/{}".format(self.label, label),
                partition_keys=self.partition_keys,
            )
            for (label, df) in new_data.items()
        }
        if is_modified:
            return self.copy(files={}, data=new_data, table_meta=new_table_meta)
        else:
            return self

    @_apply_to_list
    def apply(self, func, tables=None, metadata=None, type_safe=False):
        """
        Applies a given function to all dataframes of the MetaPartition.

        Parameters
        ----------
        func : callable or dict of callable
            A callable accepting and returning a :class:`pandas.DataFrame`
        tables : list of basestring
            Only apply and return the function on the given tables.
            Note: behavior will change in future versions!
            New behavior will be:
            Only apply the provided function to the given tables
        uuid : basestring
            The changed dataset is assigned a new UUID.
        type_safe: bool
            If the transformation is type-safe, optimizations can be applied
        Returns
        -------
        MetaPartition
            A metapartition where `func` has been applied to all internal
            DataFrames
        """
        if tables is None:
            tables = self.data.keys()
        else:
            warnings.warn(
                "The behavior for passing ``table`` parameter to ``MetaPartition.apply`` will "
                "change in the next major version. The future behavior will be to return all "
                "data and only apply the function to the selected tables. All other tables "
                "will be left untouched.",
                FutureWarning,
            )
        if callable(func):
            new_data = {k: func(v) for k, v in self.data.items() if k in tables}
        elif isinstance(func, dict):
            new_data = {k: func[k](v) for k, v in self.data.items() if k in tables}
        if metadata:
            warnings.warn(
                "The keyword argument ``metadata`` doesn't have any effect and will be removed soon.",
                DeprecationWarning,
            )
        if type_safe:
            new_table_meta = self.table_meta
        else:
            new_table_meta = {
                table: make_meta(
                    df,
                    origin="{}/{}".format(self.label, table),
                    partition_keys=self.partition_keys,
                )
                for table, df in new_data.items()
            }
        return self.copy(data=new_data, table_meta=new_table_meta)

    def as_sentinel(self):
        """
        """
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
            if "partition_keys" in kwargs:
                pk = kwargs["partition_keys"]
                return {
                    table: normalize_column_order(schema, pk)
                    for table, schema in meta.items()
                }
            else:
                return meta

        metapartitions = kwargs.get("metapartitions", None) or []
        metapartitions.extend(self.metapartitions)
        if len(metapartitions) > 1:
            first_mp = metapartitions.pop()
            mp_parent = MetaPartition(
                label=first_mp.get("label"),
                files=first_mp.get("files"),
                metadata=first_mp.get("metadata"),
                data=first_mp.get("data"),
                dataset_metadata=kwargs.get("dataset_metadata", self.dataset_metadata),
                indices=first_mp.get("indices"),
                metadata_version=self.metadata_version,
                table_meta=_renormalize_meta(kwargs.get("table_meta", self.table_meta)),
                partition_keys=kwargs.get("partition_keys", self.partition_keys),
                logical_conjunction=kwargs.get(
                    "logical_conjunction", self.logical_conjunction
                ),
            )
            for mp in metapartitions:
                mp_parent = mp_parent.add_metapartition(
                    MetaPartition(
                        label=mp.get("label"),
                        files=mp.get("files"),
                        metadata=mp.get("metadata"),
                        data=mp.get("data"),
                        dataset_metadata=mp.get(
                            "dataset_metadata", self.dataset_metadata
                        ),
                        indices=mp.get("indices"),
                        metadata_version=self.metadata_version,
                        table_meta=_renormalize_meta(
                            kwargs.get("table_meta", self.table_meta)
                        ),
                        partition_keys=kwargs.get(
                            "partition_keys", self.partition_keys
                        ),
                        logical_conjunction=kwargs.get(
                            "logical_conjunction", self.logical_conjunction
                        ),
                    ),
                    schema_validation=False,
                )
            return mp_parent
        else:
            mp = MetaPartition(
                label=kwargs.get("label", self.label),
                files=kwargs.get("files", self.files),
                data=kwargs.get("data", self.data),
                dataset_metadata=kwargs.get("dataset_metadata", self.dataset_metadata),
                indices=kwargs.get("indices", self.indices),
                metadata_version=kwargs.get("metadata_version", self.metadata_version),
                table_meta=_renormalize_meta(kwargs.get("table_meta", self.table_meta)),
                partition_keys=kwargs.get("partition_keys", self.partition_keys),
                logical_conjunction=kwargs.get(
                    "logical_conjunction", self.logical_conjunction
                ),
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
            col_in_partition = False
            for df in self.data.values():

                if col in df:
                    possible_values = possible_values | set(df[col].dropna().unique())
                    col_in_partition = True

            if (self.label is not None) and (not col_in_partition):
                raise RuntimeError(
                    "Column `{corrupt_col}` could not be found in the partition `{partition_label}` "
                    "with tables `{tables}`. Please check for any typos and validate your dataset.".format(
                        corrupt_col=col,
                        partition_label=self.label,
                        tables=sorted(self.data.keys()),
                    )
                )

            # There is at least one table with this column (see check above), so we can get the dtype from there. Also,
            # shared dtypes are ensured to be compatible.
            dtype = list(
                meta.field(col).type
                for meta in self.table_meta.values()
                if col in meta.names
            )[0]
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
        partition_on: list or str


        """
        if partition_on == self.partition_keys:
            return self

        for partition_column in partition_on:
            if partition_column in self.indices:
                raise ValueError(
                    "Trying to `partition_on` on a column with an explicit index!"
                )
        new_mp = self.as_sentinel().copy(
            partition_keys=partition_on,
            table_meta={
                table: normalize_column_order(schema, partition_on)
                for table, schema in self.table_meta.items()
            },
        )

        if isinstance(partition_on, str):
            partition_on = [partition_on]
        partition_on = self._ensure_compatible_partitioning(partition_on)

        new_data = self._partition_data(partition_on)

        for label, data_dct in new_data.items():
            tmp_mp = MetaPartition(
                label=label,
                files=self.files,
                data=data_dct,
                dataset_metadata=self.dataset_metadata,
                metadata_version=self.metadata_version,
                indices={},
                table_meta={
                    table: normalize_column_order(schema, partition_on).with_origin(
                        "{}/{}".format(table, label)
                    )
                    for table, schema in self.table_meta.items()
                },
                partition_keys=partition_on,
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
        empty_tables = []

        for table, df in self.data.items():
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

            # The handling of empty dfs is not part of the arrow implementation
            if df.empty:
                empty_tables.append((table, df))

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
                dct[new_label][table] = group
                size_after += len(group)

            if size_before != size_after:
                raise ValueError(
                    f"Original dataframe size ({size_before} rows) does not "
                    f"match new dataframe size ({size_after} rows) for table {table}. "
                    f"Hint: you may see this if you are trying to use `partition_on` on a column with null values."
                )

        for label, table_dct in dct.items():
            for empty_table, df in empty_tables:
                if empty_table not in table_dct:
                    table_dct[empty_table] = df.drop(labels=partition_on, axis=1)

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
    def _merge_metadata(metapartitions, metadata_merger=None):
        if metadata_merger is None:
            metadata_merger = combine_metadata

        new_ds_meta = metadata_merger([mp.dataset_metadata for mp in metapartitions])

        return new_ds_meta

    @staticmethod
    def merge_metapartitions(metapartitions, label_merger=None, metadata_merger=None):
        LOGGER.debug("Merging metapartitions")
        data = defaultdict(list)
        new_metadata_version = -1
        logical_conjunction = None

        for mp in metapartitions:
            new_metadata_version = max(new_metadata_version, mp.metadata_version)
            for label, df in mp.data.items():
                data[label].append(df)
            if mp.logical_conjunction or logical_conjunction:
                if logical_conjunction != mp.logical_conjunction:
                    raise TypeError(
                        "Can only merge metapartitions belonging to the same logical partition."
                    )
                else:
                    logical_conjunction = mp.logical_conjunction

        new_data = {}
        for label in data:
            if len(data[label]) == 1:
                new_data[label] = data[label][0]
            else:
                for ix, idf in enumerate(data[label]):
                    new_label = "{}_{}".format(label, ix)
                    new_data[new_label] = idf

        new_label = MetaPartition._merge_labels(metapartitions, label_merger)
        new_ds_meta = MetaPartition._merge_metadata(metapartitions, metadata_merger)

        new_mp = MetaPartition(
            label=new_label,
            data=new_data,
            dataset_metadata=new_ds_meta,
            metadata_version=new_metadata_version,
            logical_conjunction=logical_conjunction,
        )

        return new_mp

    @staticmethod
    def concat_metapartitions(metapartitions, label_merger=None, metadata_merger=None):
        LOGGER.debug("Concatenating metapartitions")
        data = defaultdict(list)
        schema = defaultdict(list)
        new_metadata_version = -1
        for mp in metapartitions:
            new_metadata_version = max(new_metadata_version, mp.metadata_version)
            for table in mp.data:
                data[table].append(mp.data[table])
                schema[table].append(mp.table_meta[table])
            # Don't care about the partition_keys. If we try to merge
            # MetaPartitions without alignment the schemas won't match.
            partition_keys = mp.partition_keys

        new_data = {}
        new_schema = {}

        for table in data:
            if len(data[table]) == 1:
                new_data[table] = data[table][0]
            else:
                new_data[table] = pd.concat(data[table])

            new_schema[table] = validate_compatible(schema[table])

        new_label = MetaPartition._merge_labels(metapartitions, label_merger)
        new_ds_meta = MetaPartition._merge_metadata(metapartitions, metadata_merger)

        new_mp = MetaPartition(
            label=new_label,
            data=new_data,
            dataset_metadata=new_ds_meta,
            metadata_version=new_metadata_version,
            table_meta=new_schema,
            partition_keys=partition_keys,
        )

        return new_mp

    @_apply_to_list
    def delete_from_store(
        self, dataset_uuid: Any, store: StoreInput
    ) -> "MetaPartition":
        store = ensure_store(store)
        # Delete data first
        for file_key in self.files.values():
            store.delete(file_key)
        return self.copy(files={}, data={}, metadata={})

    def get_parquet_metadata(self, store: StoreInput, table_name: str) -> pd.DataFrame:
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
        if not isinstance(table_name, str):
            raise TypeError("Expecting a string for parameter `table_name`.")

        store = ensure_store(store)

        data = {}
        if table_name in self.files:
            with store.open(self.files[table_name]) as fd:  # type: ignore
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


def partition_labels_from_mps(mps):
    """
    Get a list of partition labels, flattening any nested meta partitions in the input and ignoring sentinels.

    Parameters
    ----------
    mps: List[MetaPartition]

    Returns
    -------
    partition_labels: List[str]
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
    obj, metadata_version=None, expected_secondary_indices=False
) -> MetaPartition:
    """
    Parses given user input and returns a MetaPartition

    The format specification supports multiple input modes as following:

    1. Mode - Dictionary with partition information

        In this case, a dictionary is supplied where the keys describe the partition.

            * **label** - (optional) Unique partition label. If None is given, a UUID \
                        is generated using :func:`kartothek.core.uuid.gen_uuid`.
            * **data** - A dict or list of tuples. The keys represent the table name \
                        and the values are the actual payload data as a pandas.DataFrame.
            * **indices** - Deprecated, see the keyword argument `secondary_indices` to create indices.
                            A dictionary to describe the dataset indices. All \
                            partition level indices are finally merged using \
                            :func:`kartothek.io_components.metapartition.MetaPartition.merge_indices` \
                            into a single dataset index

        Examples::

            # A partition with explicit label, no metadata, one table and index information
            input_obj = {
                'label': 'partition_label',
                'data': [('table', pd.DataFrame([{'column_1':values_1, 'column_2':values_2}]))],
                'indices': {
                    "column_1": {
                        value: ['partition_label']
                    }
                }
            }
            # If no label is given, a UUID will be generated using :func:`kartothek.core.uuid.gen_uuid`
            simple_input = {
                'data': [('table', pd.DataFrame())],
            }

    2. Mode - `pandas.DataFrame`

        If only a DataFrame is provided, a UUID is generated and the dataframe is stored
        for the table name :data:`SINGLE_TABLE`

    3. Mode - :class:`~kartothek.io_components.metapartition.MetaPartition`

        If a MetaPartition is passed directly, it is simply passed through.

    4. Mode - List of tuples

        The first item represents the table name and the second is the actual payload data \
        as a pandas.DataFrame.

        Example::

            # A partition with no explicit label, no metadata and one table
            input_obj = [('table', pd.DataFrame())]

    Nested MetaPartitions:

        The input may also be provided as a list to ease batch processing. The returned MetaPartition
        will be nested and each list element represents a single physical partition. For details on
        nested MetaPartitions, see :class:`~kartothek.io_components.metapartition.MetaPartition`

    Parameters
    ----------
    obj : Union[Dict, pd.DataFrame, kartothek.io_components.metapartition.MetaPartition, List]
    metadata_version : int, optional
        The kartothek dataset specification version
    expected_secondary_indices : Optional[Union[Iterable[str], Literal[False]]]
        Iterable of strings containing expected columns on which indices are created. An empty iterable indicates no
        indices are expected.
        The default is `False`, which, indicates no checking will be done (`None` behaves the same way).
        This is only used in mode "Dictionary with partition information".

    Raises
    ------
    ValueError
        In case the given input is not understood

    Returns
    -------
    MetaPartition
    """

    if obj is None:
        obj = []
    if isinstance(obj, list):
        if len(obj) == 0:
            return MetaPartition(label=None, metadata_version=metadata_version)
        first_element = obj[0]
        if isinstance(first_element, tuple):
            data = {"data": [df] for df in obj}
            return parse_input_to_metapartition(
                obj=data,
                metadata_version=metadata_version,
                expected_secondary_indices=expected_secondary_indices,
            )
        mp = parse_input_to_metapartition(
            obj=first_element,
            metadata_version=metadata_version,
            expected_secondary_indices=expected_secondary_indices,
        )
        for mp_in in obj[1:]:
            mp = mp.add_metapartition(
                parse_input_to_metapartition(
                    obj=mp_in,
                    metadata_version=metadata_version,
                    expected_secondary_indices=expected_secondary_indices,
                )
            )
    elif isinstance(obj, dict):
        if not obj.get("data"):
            data = obj
        elif isinstance(obj["data"], list):
            data = dict(obj["data"])
        else:
            data = obj["data"]

        indices = obj.get("indices", {})
        if indices:
            warnings.warn(
                "The explicit input of indices using the `indices` key is deprecated."
                'Use the `secondary_indices` keyword argument of "write" and "update" functions instead.',
                DeprecationWarning,
            )
        indices = {k: v for k, v in indices.items() if v}
        _ensure_valid_indices(
            mp_indices=indices, secondary_indices=expected_secondary_indices, data=data
        )

        mp = MetaPartition(
            # TODO: Deterministic hash for the input?
            label=obj.get("label", gen_uuid()),
            data=data,
            indices=indices,
            metadata_version=metadata_version,
        )
    elif isinstance(obj, pd.DataFrame):
        mp = MetaPartition(
            label=gen_uuid(),
            data={SINGLE_TABLE: obj},
            metadata_version=metadata_version,
        )
    elif isinstance(obj, MetaPartition):
        return obj
    else:
        raise ValueError("Unexpected type: {}".format(type(obj)))

    return mp
