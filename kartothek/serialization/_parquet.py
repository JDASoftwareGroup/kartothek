#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functionality for persisting/serialising DataFrames.
"""


import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from simplekv import KeyValueStore

from ._generic import (
    DataFrameSerializer,
    PredicatesType,
    check_predicates,
    filter_df,
    filter_df_from_predicates,
)
from ._io_buffer import BlockBuffer
from ._util import ensure_unicode_string_type

try:
    # Only check for BotoStore instance if boto is really installed
    from simplekv.net.botostore import BotoStore

    HAVE_BOTO = True
except ImportError:
    HAVE_BOTO = False


EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()


def _empty_table_from_schema(parquet_file):
    schema = parquet_file.schema.to_arrow_schema()

    return schema.empty_table()


def _reset_dictionary_columns(table, exclude=None):
    """
    We need to ensure that the dtype is exactly as requested, see GH227
    """
    if exclude is None:
        exclude = []

    schema = table.schema
    for i in range(len(schema)):
        field = schema[i]
        if field.name in exclude:
            continue
        if pa.types.is_dictionary(field.type):
            new_field = pa.field(
                field.name, field.type.value_type, field.nullable, field.metadata,
            )
            schema = schema.remove(i).insert(i, new_field)

    table = table.cast(schema)
    return table


class ParquetSerializer(DataFrameSerializer):
    _PARQUET_VERSION = "2.0"
    type_stable = True

    def __init__(self, compression="SNAPPY", chunk_size=None):
        self.compression = compression
        self.chunk_size = chunk_size

    def __eq__(self, other):
        return (
            isinstance(other, ParquetSerializer)
            and (self.compression == other.compression)
            and (self.chunk_size == other.chunk_size)
        )

    def __repr__(self):
        return "ParquetSerializer(compression={compression!r}, chunk_size={chunk_size!r})".format(
            compression=self.compression, chunk_size=self.chunk_size
        )

    @staticmethod
    def restore_dataframe(
        store: KeyValueStore,
        key: str,
        filter_query: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
        predicate_pushdown_to_io: bool = True,
        categories: Optional[Iterable[str]] = None,
        predicates: Optional[PredicatesType] = None,
        date_as_object: bool = False,
    ):
        check_predicates(predicates)
        # If we want to do columnar access we can benefit from partial reads
        # otherwise full read en block is the better option.
        if (not predicate_pushdown_to_io) or (columns is None and predicates is None):
            with pa.BufferReader(store.get(key)) as reader:
                table = pq.read_pandas(reader, columns=columns)
        else:
            if HAVE_BOTO and isinstance(store, BotoStore):
                # Parquet and seeks on S3 currently leak connections thus
                # we omit column projection to the store.
                reader = pa.BufferReader(store.get(key))
            else:
                reader = store.open(key)
                # Buffer at least 4 MB in requests. This is chosen because the default block size of the Azure
                # storage client is 4MB.
                reader = BlockBuffer(reader, 4 * 1024 * 1024)
            try:
                parquet_file = ParquetFile(reader)
                if predicates and parquet_file.metadata.num_rows > 0:
                    # We need to calculate different predicates for predicate
                    # pushdown and the later DataFrame filtering. This is required
                    # e.g. in the case where we have an `in` predicate as this has
                    # different normalized values.
                    columns_to_io = _columns_for_pushdown(columns, predicates)
                    predicates_for_pushdown = _normalize_predicates(
                        parquet_file, predicates, True
                    )
                    predicates = _normalize_predicates(parquet_file, predicates, False)
                    tables = _read_row_groups_into_tables(
                        parquet_file, columns_to_io, predicates_for_pushdown
                    )

                    if len(tables) == 0:
                        table = _empty_table_from_schema(parquet_file)
                    else:
                        table = pa.concat_tables(tables)
                else:
                    # ARROW-5139 Column projection with empty columns returns a table w/out index
                    if columns == []:
                        # Create an arrow table with expected index length.
                        df = (
                            parquet_file.schema.to_arrow_schema()
                            .empty_table()
                            .to_pandas(date_as_object=date_as_object)
                        )
                        index = pd.Int64Index(
                            pd.RangeIndex(start=0, stop=parquet_file.metadata.num_rows)
                        )
                        df = pd.DataFrame(df, index=index)
                        # convert back to table to keep downstream code untouched by this patch
                        table = pa.Table.from_pandas(df)
                    else:
                        table = pq.read_pandas(reader, columns=columns)
            finally:
                reader.close()

        if columns is not None:
            missing_columns = set(columns) - set(table.schema.names)
            if missing_columns:
                raise ValueError(
                    "Columns cannot be found in stored dataframe: {missing}".format(
                        missing=", ".join(sorted(missing_columns))
                    )
                )

        table = _reset_dictionary_columns(table, exclude=categories)
        df = table.to_pandas(categories=categories, date_as_object=date_as_object)
        df.columns = df.columns.map(ensure_unicode_string_type)
        if predicates:
            df = filter_df_from_predicates(
                df, predicates, strict_date_types=date_as_object
            )
        else:
            df = filter_df(df, filter_query)
        if columns is not None:
            return df.reindex(columns=columns, copy=False)
        else:
            return df

    def store(self, store, key_prefix, df):
        key = "{}.parquet".format(key_prefix)
        if isinstance(df, pa.Table):
            table = df
        else:
            table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()

        pq.write_table(
            table,
            buf,
            version=self._PARQUET_VERSION,
            chunk_size=self.chunk_size,
            compression=self.compression,
            coerce_timestamps="us",
        )
        store.put(key, buf.getvalue().to_pybytes())
        return key


def _columns_for_pushdown(columns, predicates):
    if columns is None:
        return
    new_cols = columns[:]
    for conjunction in predicates:
        for literal in conjunction:
            if literal[0] not in columns:
                new_cols.append(literal[0])
    return new_cols


def _read_row_groups_into_tables(parquet_file, columns, predicates_in):
    """
    For each RowGroup check if the predicate in DNF applies and then
    read the respective RowGroup.
    """
    arrow_schema = parquet_file.schema.to_arrow_schema()
    parquet_reader = parquet_file.reader

    def all_predicates_accept(row):
        # Check if the predicates evaluate on this RowGroup.
        # As the predicate is in DNF, we only need a single of the
        # inner lists to match. Once we have found a positive match,
        # there is no need to check whether the remaining ones apply.
        row_meta = parquet_file.metadata.row_group(row)
        for predicate_list in predicates_in:
            if all(
                _predicate_accepts(predicate, row_meta, arrow_schema, parquet_reader)
                for predicate in predicate_list
            ):
                return True
        return False

    # Iterate over the RowGroups and evaluate the list of predicates on each
    # one of them. Only access those that could contain a row where we could
    # get an exact match of the predicate.
    result = []
    for row in range(parquet_file.num_row_groups):
        if all_predicates_accept(row):
            row_group = parquet_file.read_row_group(row, columns=columns)
            result.append(row_group)
    return result


def _normalize_predicates(parquet_file, predicates, for_pushdown):
    schema = parquet_file.schema.to_arrow_schema()

    normalized_predicates = []
    for conjunction in predicates:
        new_conjunction = []

        for literal in conjunction:
            col, op, val = literal
            col_idx = parquet_file.reader.column_name_idx(col)
            pa_type = schema[col_idx].type
            column_name = schema[col_idx].name

            if pa.types.is_null(pa_type):
                # early exit, the entire conjunction evaluates to False
                new_conjunction = None
                break

            if op == "in":
                values = [
                    _normalize_value(lit, pa_type, column_name=column_name)
                    for lit in literal[2]
                ]
                if for_pushdown and values:
                    normalized_value = [
                        _timelike_to_arrow_encoding(value, pa_type) for value in values
                    ]
                else:
                    normalized_value = values
            else:
                normalized_value = _normalize_value(
                    literal[2], pa_type, column_name=column_name
                )
                if for_pushdown:
                    normalized_value = _timelike_to_arrow_encoding(
                        normalized_value, pa_type
                    )
            new_literal = (literal[0], literal[1], normalized_value)
            new_conjunction.append(new_literal)

        if new_conjunction is not None:
            normalized_predicates.append(new_conjunction)
    return normalized_predicates


def _timelike_to_arrow_encoding(value, pa_type):
    # Date32 columns are encoded as days since 1970
    if pa.types.is_date32(pa_type):
        if isinstance(value, datetime.date):
            return value.toordinal() - EPOCH_ORDINAL
    else:
        return value


def _normalize_value(value, pa_type, column_name=None):
    if pa.types.is_dictionary(pa_type):
        pa_type = pa_type.value_type

    if pa.types.is_string(pa_type):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        elif isinstance(value, str):
            return value
        elif value is None:
            return value
    elif pa.types.is_binary(pa_type):
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return str(value).encode("utf-8")
    elif (
        pa.types.is_integer(pa_type)
        and pd.api.types.is_integer(value)
        or pa.types.is_floating(pa_type)
        and pd.api.types.is_float(value)
        or pa.types.is_boolean(pa_type)
        and pd.api.types.is_bool(value)
        or pa.types.is_timestamp(pa_type)
        and not isinstance(value, (bytes, str))
        and (
            pd.api.types.is_datetime64_dtype(value)
            or isinstance(value, datetime.datetime)
        )
    ):
        return value
    elif pa.types.is_date(pa_type):
        if isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        elif isinstance(value, bytes):
            value = value.decode("utf-8")
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        elif isinstance(value, datetime.date):
            if isinstance(value, datetime.datetime):
                raise TypeError(
                    f"Unexpected type for predicate: Column {column_name!r} is an "
                    f"Arrow date ({pa_type}), but predicate value has type {type(value)}. "
                    f"Use a Python 'datetime.date' object instead."
                )
            else:
                return value
    predicate_value_dtype = pd.Series(value).dtype
    raise TypeError(
        f"Unexpected type for predicate: Column {column_name!r} has pandas type "
        f"{pa_type.to_pandas_dtype()} (Arrow type {pa_type}), but predicate value "
        f"{value!r} has pandas type '{predicate_value_dtype}' (Python type '{type(value)}')"
    )


def _predicate_accepts(predicate, row_meta, arrow_schema, parquet_reader):
    """
    Checks if a predicate evaluates on a column.

    This method first casts the value of the predicate to the type used for this column
    in the statistics and then applies the relevant operator. The operation applied here
    is done in a fashion to check if the predicate would evaluate to True for any possible
    row in the RowGroup. Thus e.g. for the `==` predicate, we check if the predicate value
    is in the (min, max) range of the RowGroup.
    """
    col, op, val = predicate
    col_idx = parquet_reader.column_name_idx(col)
    pa_type = arrow_schema[col_idx].type
    parquet_statistics = row_meta.column(col_idx).statistics

    # In case min/max is not set, we have to assume that the predicate matches.
    if not parquet_statistics.has_min_max:
        return True

    min_value = parquet_statistics.min
    max_value = parquet_statistics.max
    # Transform the predicate value to the respective type used in the statistics.

    # integer overflow protection since statistics are stored as signed integer, see ARROW-5166
    if pa.types.is_integer(pa_type) and (max_value < min_value):
        return True

    if pa.types.is_timestamp(pa_type):
        # timestamps in the parquet statistic might be of type datetime.datetime, which is not compatible w/ numpy
        min_value = np.datetime64(min_value)
        max_value = np.datetime64(max_value)

    # The statistics for floats only contain the 6 most significant digits.
    # So a suitable epsilon has to be considered below min and above max.
    if isinstance(val, float):
        min_value -= _epsilon(min_value)
        max_value += _epsilon(max_value)

    # op can only be "==" or "!=" for scalar null values.
    if op == "==":
        if pd.isnull(val):
            return parquet_statistics.null_count > 0
        else:
            return (min_value <= val) and (max_value >= val)
    elif op == "!=":
        if pd.isnull(val):
            return parquet_statistics.null_count < row_meta.num_rows
        else:
            return not ((min_value >= val) and (max_value <= val))
    elif op == "<=":
        return min_value <= val
    elif op == ">=":
        return max_value >= val
    elif op == "<":
        return min_value < val
    elif op == ">":
        return max_value > val
    elif op == "in":
        # This implementation is chosen for performance reasons. See
        # https://github.com/JDASoftwareGroup/kartothek/pull/130 for more information/benchmarks.
        # We accept the predicate if there is any value in the provided array which is equal to or between
        # the parquet min and max statistics. Otherwise, it is rejected.
        for x in val:
            if pd.isnull(x):
                if parquet_statistics.null_count > 0:
                    return True
            elif min_value <= x <= max_value:
                return True
        return False
    else:
        raise NotImplementedError("op not supported")


def _highest_significant_position(num):
    """
    >>> _highest_significant_position(1.0)
    1
    >>> _highest_significant_position(9.0)
    1
    >>> _highest_significant_position(39.0)
    2
    >>> _highest_significant_position(0.1)
    -1
    >>> _highest_significant_position(0.9)
    -1
    >>> _highest_significant_position(0.000123)
    -4
    >>> _highest_significant_position(1234567.0)
    7
    >>> _highest_significant_position(-0.1)
    -1
    >>> _highest_significant_position(-100.0)
    3
    """
    abs_num = np.absolute(num)
    log_of_abs = np.log10(abs_num)
    position = int(np.floor(log_of_abs))

    # is position left of decimal point?
    if abs_num >= 1.0:
        position += 1

    return position


def _epsilon(num):
    """
    >>> _epsilon(123456)
    1
    >>> _epsilon(0.123456)
    1e-06
    >>> _epsilon(0.123)
    1e-06
    >>> _epsilon(0)
    0
    >>> _epsilon(-0.123456)
    1e-06
    >>> _epsilon(-123456)
    1
    >>> _epsilon(np.inf)
    0
    >>> _epsilon(-np.inf)
    0
    """
    SIGNIFICANT_DIGITS = 6

    if num == 0 or np.isinf(num):
        return 0

    epsilon_position = _highest_significant_position(num) - SIGNIFICANT_DIGITS

    # is position right of decimal point?
    if epsilon_position < 0:
        epsilon_position += 1

    return 10 ** epsilon_position
