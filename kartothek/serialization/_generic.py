#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functionality for persisting/serialising DataFrames.
"""

from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from kartothek.serialization._util import _check_contains_null

from ._util import ensure_unicode_string_type


class DataFrameSerializer:
    """
    Abstract class that supports serializing DataFrames to/from
    simplekv stores.
    """

    _serializers: Dict[str, "DataFrameSerializer"] = {}
    type_stable = False

    def __ne__(self, other):
        return not (self == other)

    @classmethod
    def register_serializer(cls, suffix, serializer):
        cls._serializers[suffix] = serializer

    @classmethod
    def restore_dataframe(
        cls,
        store,
        key,
        filter_query=None,
        columns=None,
        predicate_pushdown_to_io=True,
        categories=None,
        predicates=None,
        date_as_object=False,
    ):
        """
        Load a DataFrame from the specified store. The key is also used to
        detect the used format.

        Parameters
        ----------
        store: simplekv.KeyValueStore
                store engine
        key: str
                Key that specifies a path where object should be
                retrieved from the store resource.
        filter_query: str
                Optional query to filter the DataFrame. Must adhere to the specification
                of pandas.DataFrame.query.
        columns : str or None
                Only read in listed columns. When set to None, the full file
                will be read in.
        predicate_pushdown_to_io: bool
                Push predicates through to the I/O layer, default True. Disable
                this if you see problems with predicate pushdown for the given
                file even if the file format supports it. Note that this option
                only hides problems in the store layer that need to be addressed
                there.
        categories: list of str (optional)
                Columns that should be loaded as categoricals.
        predicates: list of list of tuple[str, str, Any]
                Optional list of predicates, like [[('x', '>', 0), ...], that are used
                to filter the resulting DataFrame, possibly using predicate pushdown,
                if supported by the file format.
                This parameter is not compatible with filter_query.

                Predicates are expressed in disjunctive normal form (DNF). This means
                that the innermost tuple describe a single column predicate. These
                inner predicate make are all combined with a conjunction (AND) into a
                larger predicate. The most outer list then combines all predicates
                with a disjunction (OR). By this, we should be able to express all
                kinds of predicates that are possible using boolean logic.
        date_as_object: bool
                Retrieve all date columns as an object column holding datetime.date objects
                instead of pd.Timestamp. Note that this option only works for type-stable
                serializers, e.g. ``ParquetSerializer``.
        Returns
        -------
        Data in pandas dataframe format.
        """
        if filter_query and predicates:
            raise ValueError("Can only specify one of filter_query and predicates")

        for suffix, serializer in cls._serializers.items():
            if key.endswith(suffix):
                df = serializer.restore_dataframe(
                    store,
                    key,
                    filter_query,
                    columns,
                    predicate_pushdown_to_io=predicate_pushdown_to_io,
                    categories=categories,
                    predicates=predicates,
                    date_as_object=date_as_object,
                )
                df.columns = df.columns.map(ensure_unicode_string_type)
                return df

        # No serialiser matched
        raise ValueError(
            "The specified file format for '{}' is not supported".format(key)
        )

    def store(self, store, key_prefix, df):
        """
        Persist a DataFrame to the specified store.

        The used store format (e.g. Parquet) will be appended to the key.

        Parameters
        ----------
        store: simplekv.KeyValueStore
                store engine
        key_prefix: str
                Key prefix that specifies a path where object should be
                stored on the store resource. The used file format will be
                appended to the key.
        df: pandas.DataFrame or pyarrow.Table
                DataFrame that shall be persisted

        Returns
        -------
        str
            The actual key where the DataFrame is stored.
        """
        raise NotImplementedError("Abstract method called.")


def filter_df(df, filter_query=None):
    """
    General implementation of query filtering.

    Serialisation formats such as Parquet that support predicate push-down
    may pre-filter in their own implementations.
    """
    if df.shape[0] > 0 and filter_query is not None:
        df = df.query(filter_query)
    return df


def check_predicates(predicates):
    """
    Check if predicates are well-formed.
    """
    if predicates is not None:
        if len(predicates) == 0 or any(len(p) == 0 for p in predicates):
            raise ValueError("Malformed predicates")
        for conjunction in predicates:
            for col, op, val in conjunction:
                if (
                    isinstance(val, list)
                    and any(_check_contains_null(v) for v in val)
                    or _check_contains_null(val)
                ):
                    raise NotImplementedError(
                        "Null-terminated binary strings are not supported as predicate values."
                    )


def filter_df_from_predicates(df, predicates, strict_date_types=False):
    """
    Filter a `pandas.DataFrame` based on predicates in disjunctive normal form.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame to be filtered
    predicates: list of lists
        Predicates in disjunctive normal form (DNF). For a thorough documentation, see
        :class:`DataFrameSerializer.restore_dataframe`
    strict_date_types: bool
        If False (default), cast all datelike values to datetime64 for comparison.

    Returns
    -------
    pd.DataFrame
    """
    indexer = np.zeros(len(df), dtype=bool)
    for conjunction in predicates:
        inner_indexer = np.ones(len(df), dtype=bool)
        for column, op, value in conjunction:
            filter_array_like(
                df[ensure_unicode_string_type(column)].values,
                op,
                value,
                inner_indexer,
                inner_indexer,
                strict_date_types=strict_date_types,
            )
        indexer = inner_indexer | indexer
    return df[indexer]


def _handle_categorical_data(array_like, require_ordered):
    if require_ordered and pd.api.types.is_categorical(array_like):
        if isinstance(array_like, pd.Categorical):
            categorical = array_like
        else:
            categorical = array_like.cat
        array_value_type = categorical.categories.dtype
        if categorical.categories.is_monotonic:
            array_like = categorical.as_ordered()
        else:
            array_like = categorical.reorder_categories(
                categorical.categories.sort_values(), ordered=True
            )
    else:
        array_value_type = array_like.dtype
    return array_like, array_value_type


def _handle_null_arrays(array_like, value_dtype):
    # NULL types might not be preserved well, so try to cast floats (pandas default type) to the value type
    # Determine the type using the `kind` interface since this is common for a numpy array, pandas series and pandas extension arrays
    if array_like.dtype.kind == "f" and np.isnan(array_like).all():
        if array_like.dtype.kind != value_dtype.kind:
            array_like = array_like.astype(value_dtype)
    return array_like, array_like.dtype


def _handle_timelike_values(array_value_type, value, value_dtype, strict_date_types):
    if is_list_like(value):
        value = [pd.Timestamp(val).to_datetime64() for val in value]
    else:
        value = pd.Timestamp(value).to_datetime64()
    value_dtype = pd.Series(value).dtype
    return value, value_dtype


def _ensure_type_stability(array_like, value, strict_date_types, require_ordered):
    """
    Ensure that the provided value and the provided array will have compatible
    types, such that comparisons are unambiguous.

    The type check is based on the numpy type system and accesses the arrays
    `kind` attribute and asserts equality. The provided value will be
    interpreted as a scalar in this case. For scalars which do not have a proper
    python representation, we will relax the strictness as long as there is a
    valid and unambiguous interpretation of a comparison operation. In
    particular we consider the following combinations valid:

        * unsigned integer (u) <> integer (i)
        * zero-terminated bytes (S) <> Python Object (O)
        * Unicode string (U) <> Python Object (O)

    Parameters
    ----------
    strict_date_types: bool
        If False, assume that datetime.date and datetime.datetime are
        compatible types. In this case, the value is cast appropriately
    require_ordered: bool
        Indicate if the operator to be evaluated will require a notion of
        ordering. In the case of pd.Categorical we will then assume a
        lexicographical ordering and cast the pd.CategoricalDtype accordingly
    """

    value_dtype = pd.Series(value).dtype
    array_like, array_value_type = _handle_categorical_data(array_like, require_ordered)
    array_like, array_value_type = _handle_null_arrays(array_like, value_dtype)

    type_comp = {value_dtype.kind, array_value_type.kind}

    compatible_types = [
        # UINT and INT
        {"u", "i"},
        # various string kinds
        {"S", "O"},
        {"U", "O"},
    ]

    if not strict_date_types:
        # objects (datetime.date) and datetime64
        compatible_types.append({"O", "M"})

    type_comp = {value_dtype.kind, array_value_type.kind}

    if len(type_comp) > 1 and type_comp not in compatible_types:
        raise TypeError(
            f"Unexpected type encountered. Expected {array_value_type.kind} but got {value_dtype.kind}."
        )
    if "M" in type_comp:
        value, value_dtype = _handle_timelike_values(
            array_value_type, value, value_dtype, strict_date_types
        )
    return array_like, value


def filter_array_like(
    array_like, op, value, mask=None, out=None, strict_date_types=False
):
    """
    Filter an array-like object using operations defined in the predicates

    Parameters
    ----------
    array_like: array-like, c.f. pd.api.types.is_array_like
        The array like object to be filtered
    op: string
    value: object
    mask: boolean array-like, optional
        A boolean array like object which will be combined with the result
        of this evaluation using a logical AND. If an array with all True is
        given, it will be the same result as if left empty
    out: array-like
        An array into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated
        array is returned.
    strict_date_types: bool
        If False (default), cast all datelike values to datetime64 for comparison.
    """
    if mask is None:
        mask = np.ones(len(array_like), dtype=bool)

    if out is None:
        out = np.zeros(len(array_like), dtype=bool)

    # In the case of an empty list, don't bother with evaluating types, etc.
    if is_list_like(value) and len(value) == 0:
        false_arr = np.zeros(len(array_like), dtype=bool)
        np.logical_and(false_arr, mask, out=out)
        return out

    require_ordered = "<" in op or ">" in op
    array_like, value = _ensure_type_stability(
        array_like, value, strict_date_types, require_ordered
    )

    with np.errstate(invalid="ignore"):
        if op == "==":
            np.logical_and(array_like == value, mask, out=out)
        elif op == "!=":
            np.logical_and(array_like != value, mask, out=out)
        elif op == "<=":
            np.logical_and(array_like <= value, mask, out=out)
        elif op == ">=":
            np.logical_and(array_like >= value, mask, out=out)
        elif op == "<":
            np.logical_and(array_like < value, mask, out=out)
        elif op == ">":
            np.logical_and(array_like > value, mask, out=out)
        elif op == "in":
            value = np.asarray(value)
            np.logical_and(
                np.isin(array_like, value)
                if len(value) > 0
                else np.zeros(len(array_like), dtype=bool),
                mask,
                out=out,
            )
        else:
            raise NotImplementedError("op not supported")

    return out
