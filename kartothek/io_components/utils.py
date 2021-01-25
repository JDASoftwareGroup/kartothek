"""
This module is a collection of helper functions
"""
import collections
import inspect
import logging
from typing import List, Optional, TypeVar, Union, overload

import decorator
import pandas as pd

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.factory import _ensure_factory
from kartothek.core.typing import StoreFactory, StoreInput
from kartothek.core.utils import ensure_store, lazy_store

try:
    from typing_extensions import Literal  # type: ignore
except ImportError:
    from typing import Literal  # type: ignore


signature = inspect.signature


LOGGER = logging.getLogger(__name__)


class InvalidObject:
    """
    Sentinel to mark keys for removal
    """

    pass


def combine_metadata(dataset_metadata, append_to_list=True):
    """
    Merge a list of dictionaries

    The merge is performed in such a way, that only keys which
    are present in **all** dictionaries are kept in the final result.

    If lists are encountered, the values of the result will be the
    concatenation of all list values in the order of the supplied dictionary list.
    This behaviour may be changed by using append_to_list

    Parameters
    ----------
    dataset_metadata: list of dict
        The list of dictionaries (usually metadata) to be combined.
    append_to_list: bool
        If True, all values are concatenated. If False, only unique values are kept
    """
    meta = _combine_metadata(dataset_metadata, append_to_list)
    return _remove_invalids(meta)


def _remove_invalids(dct):
    if not isinstance(dct, dict):
        return {}

    new_dict = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            tmp = _remove_invalids(value)
            # Do not propagate empty dicts
            if tmp:
                new_dict[key] = tmp
        elif not isinstance(value, InvalidObject):
            new_dict[key] = value
    return new_dict


def _combine_metadata(dataset_metadata, append_to_list):
    assert isinstance(dataset_metadata, list)
    if len(dataset_metadata) == 1:
        return dataset_metadata.pop()

    # In case the input list has only two elements, we can do simple comparison
    if len(dataset_metadata) > 2:
        first = _combine_metadata(dataset_metadata[::2], append_to_list)
        second = _combine_metadata(dataset_metadata[1::2], append_to_list)
        final = _combine_metadata([first, second], append_to_list)
        return final
    else:
        first = dataset_metadata.pop()
        second = dataset_metadata.pop()
        if first == second:
            return first
        # None is harmless and may occur if a key appears in one but not the other dict
        elif first is None or second is None:
            return first if first is not None else second
        elif isinstance(first, dict) and isinstance(second, dict):
            new_dict = {}
            keys = set(first.keys())
            keys.update(second.keys())
            for key in keys:
                new_dict[key] = _combine_metadata(
                    [first.get(key), second.get(key)], append_to_list
                )
            return new_dict
        elif isinstance(first, list) and isinstance(second, list):
            new_list = first.extend(second)
            if append_to_list:
                return new_list
            else:
                return list(set(new_list))
        else:
            return InvalidObject()


def _ensure_compatible_indices(dataset, secondary_indices):
    if dataset:
        ds_secondary_indices = list(dataset.secondary_indices.keys())

        if secondary_indices and set(ds_secondary_indices) != set(secondary_indices):
            raise ValueError(
                f"Incorrect indices provided for dataset.\n"
                f"Expected: {ds_secondary_indices}\n"
                f"But got: {secondary_indices}"
            )
        return ds_secondary_indices
    else:
        # We return `False` if there is no dataset in storage and `secondary_indices` is undefined
        # (`secondary_indices` is normalized to `[]` by default).
        # In consequence, `parse_input_to_metapartition` will not check indices at the partition level.
        return secondary_indices or False


def _ensure_valid_indices(mp_indices, secondary_indices=None, data=None):
    # TODO (Kshitij68): Behavior is closely matches `_ensure_compatible_indices`. Refactoring can prove to be helpful
    if data:
        for table_name in data:
            for index in mp_indices.keys():
                if index not in data[table_name].columns:
                    raise ValueError(
                        f"In table {table_name}, no column corresponding to index {index}"
                    )
    if secondary_indices not in (False, None):
        secondary_indices = set(secondary_indices)
        # If the dataset has `secondary_indices` defined, then these indices will be build later so there is no need to
        # ensure that they are also defined here (on a partition level).
        # Hence,  we just check that no new indices are defined on the partition level.
        if not secondary_indices.issuperset(mp_indices.keys()):
            raise ValueError(
                "Incorrect indices provided for dataset.\n"
                f"Expected index columns: {secondary_indices}"
                f"Provided index: {mp_indices}"
            )


def validate_partition_keys(
    dataset_uuid,
    store,
    ds_factory,
    default_metadata_version,
    partition_on,
    load_dataset_metadata=True,
):
    if ds_factory or DatasetMetadata.exists(dataset_uuid, ensure_store(store)):
        ds_factory = _ensure_factory(
            dataset_uuid=dataset_uuid,
            store=store,
            factory=ds_factory,
            load_dataset_metadata=load_dataset_metadata,
        )

        ds_metadata_version = ds_factory.metadata_version
        if partition_on:
            if not isinstance(partition_on, list):
                partition_on = [partition_on]
            if partition_on != ds_factory.partition_keys:
                raise ValueError(
                    "Incompatible set of partition keys encountered. "
                    "Input partitioning was `{}` while actual dataset was `{}`".format(
                        partition_on, ds_factory.partition_keys
                    )
                )
        else:
            partition_on = ds_factory.partition_keys
    else:
        ds_factory = None
        ds_metadata_version = default_metadata_version
    return ds_factory, ds_metadata_version, partition_on


_NORMALIZE_ARGS_LIST = [
    "partition_on",
    "delete_scope",
    "secondary_indices",
    "sort_partitions_by",
    "bucket_by",
]

_NORMALIZE_ARGS = _NORMALIZE_ARGS_LIST + ["store", "dispatch_by"]

T = TypeVar("T")


@overload
def normalize_arg(
    arg_name: Literal[
        "partition_on",
        "delete_scope",
        "secondary_indices",
        "bucket_by",
        "sort_partitions_by",
        "dispatch_by",
    ],
    old_value: Optional[Union[T, List[T]]],
) -> List[T]:
    ...


@overload
def normalize_arg(
    arg_name: Literal["store"], old_value: Optional[StoreInput]
) -> StoreFactory:
    ...


def normalize_arg(arg_name, old_value):
    """
    Normalizes an argument according to pre-defined types

    Type A:

    * "partition_on"
    * "delete_scope"
    * "secondary_indices"
    * "dispatch_by"

    will be converted to a list. If it is None, an empty list will be created

    Type B:
    * "store"

    Will be converted to a callable returning
    """

    def _make_list(_args):
        if isinstance(_args, (str, bytes, int, float)):
            return [_args]
        if _args is None:
            return []
        if isinstance(_args, (set, frozenset, dict)):
            raise ValueError(
                "{} is incompatible for normalisation.".format(type(_args))
            )
        return list(_args)

    if arg_name in _NORMALIZE_ARGS_LIST:
        if old_value is None:
            return []
        elif isinstance(old_value, list):
            return old_value
        else:
            return _make_list(old_value)
    elif arg_name == "dispatch_by":
        if old_value is None:
            return old_value
        elif isinstance(old_value, list):
            return old_value
        else:
            return _make_list(old_value)
    elif arg_name == "store" and old_value is not None:
        return lazy_store(old_value)

    return old_value


@decorator.decorator
def normalize_args(function, *args, **kwargs):
    sig = signature(function)

    def _wrapper(*args, **kwargs):
        for arg_name in _NORMALIZE_ARGS:
            if arg_name in sig.parameters.keys():

                ix = inspect.getfullargspec(function).args.index(arg_name)
                if arg_name in kwargs:
                    kwargs[arg_name] = normalize_arg(arg_name, kwargs[arg_name])
                elif len(args) > ix:
                    new_args = list(args)
                    new_args[ix] = normalize_arg(arg_name, args[ix])
                    args = tuple(new_args)
                else:
                    kwargs[arg_name] = normalize_arg(arg_name, None)
        return function(*args, **kwargs)

    return _wrapper(*args, **kwargs)


def extract_duplicates(lst):
    """
    Return all items of a list that occur more than once.

    Parameters
    ----------
    lst: List[Any]

    Returns
    -------
    lst: List[Any]
    """

    return [item for item, count in collections.Counter(lst).items() if count > 1]


def align_categories(dfs, categoricals):
    """
    Takes a list of dataframes with categorical columns and determines the superset
    of categories. All specified columns will then be cast to the same `pd.CategoricalDtype`

    Parameters
    ----------
    dfs: List[pd.DataFrame]
        A list of dataframes for which the categoricals should be aligned
    categoricals: List[str]
        Columns holding categoricals which should be aligned
    Returns
    -------
    List[pd.DataFrame]
        A list with aligned dataframes
    """
    col_dtype = {}

    for column in categoricals:
        position_largest_df = None
        categories = set()
        largest_df_categories = set()
        for ix, df in enumerate(dfs):
            ser = df[column]
            if not pd.api.types.is_categorical_dtype(ser):
                cats = ser.unique()
                LOGGER.info(
                    "Encountered non-categorical type where categorical was expected\n"
                    "Found at index position {ix} for column {col}\n"
                    "Dtypes: {dtypes}".format(ix=ix, col=column, dtypes=df.dtypes)
                )
            else:
                cats = ser.cat.categories
                length = len(df)
                if position_largest_df is None or length > position_largest_df[0]:
                    position_largest_df = (length, ix)
                if position_largest_df[1] == ix:
                    largest_df_categories = cats
            categories.update(cats)

        # use the categories of the largest DF as a baseline to avoid having
        # to rewrite its codes. Append the remainder and sort it for reproducibility
        categories = list(largest_df_categories) + sorted(
            set(categories) - set(largest_df_categories)
        )
        cat_dtype = pd.api.types.CategoricalDtype(categories, ordered=False)
        col_dtype[column] = cat_dtype

    return_dfs = []
    for df in dfs:
        try:
            new_df = df.astype(col_dtype, copy=False)
        except ValueError as verr:
            cat_types = {
                col: dtype.categories.dtype for col, dtype in col_dtype.items()
            }
            # Should be fixed by pandas>=0.24.0
            if "buffer source array is read-only" in str(verr):
                new_df = df.astype(cat_types)
                new_df = new_df.astype(col_dtype)
            else:
                raise verr
        return_dfs.append(new_df)
    return return_dfs


def sort_values_categorical(df: pd.DataFrame, columns: Union[List[str], str]):
    """
    Sort a dataframe lexicographically by the categories of column `column`
    """
    if not isinstance(columns, list):
        columns = [columns]
    for col in columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            cat_accesor = df[col].cat
            df[col] = cat_accesor.reorder_categories(
                sorted(cat_accesor.categories), ordered=True
            )
    return df.sort_values(by=columns).reset_index(drop=True)


def check_single_table_dataset(dataset, expected_table=None):
    """
    Raise if the given dataset is not a single-table dataset.

    Parameters
    ----------
    dataset: kartothek.core.dataset.DatasetMetadata
        The dataset to be validated
    expected_table: Optional[str]
        Ensure that the table in the dataset is the same as the given one.
    """

    if len(dataset.tables) > 1:
        raise TypeError(
            "Expected single table dataset but found dataset with tables: `{}`".format(
                dataset.tables
            )
        )
    if expected_table and dataset.tables != [expected_table]:
        raise TypeError(
            "Unexpected table in dataset:\nFound:\t{}\nExpected:\t{}".format(
                dataset.tables, expected_table
            )
        )


def raise_if_indices_overlap(partition_on, secondary_indices):
    partition_secondary_overlap = set(partition_on) & set(secondary_indices)
    if partition_secondary_overlap:
        raise RuntimeError(
            f"Cannot create secondary index on partition columns: {partition_secondary_overlap}"
        )
