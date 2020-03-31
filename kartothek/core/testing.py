# -*- coding: utf-8 -*-


import contextlib
import datetime
from datetime import date
from unittest import mock
from warnings import catch_warnings, simplefilter

import hypothesis.extra.numpy as hyp_np
import hypothesis.strategies as hyp_st
import numpy as np
import pandas as pd
from hypothesis.errors import NonInteractiveExampleWarning

from kartothek.core.uuid import gen_uuid_object

TIME_TO_FREEZE = datetime.datetime(2000, 1, 1, 1, 1, 1, 1)
TIME_TO_FREEZE_ISO = "2000-01-01T01:01:01.000001"
TIME_TO_FREEZE_ISO_QUOTED = "2000-01-01T01%3A01%3A01.000001"


def get_dataframe_alltypes():
    """
    Return a pandas DataFrame of length one with a column for each commonly used data types
    """
    # fmt: off
    not_nested = get_dataframe_not_nested()
    nested_types = pd.DataFrame(
        {
            "array_int8": pd.Series([np.array([1], dtype=np.int8)], dtype=object),
            "array_int16": pd.Series([np.array([1], dtype=np.int16)], dtype=object),
            "array_int32": pd.Series([np.array([1], dtype=np.int32)], dtype=object),
            "array_int64": pd.Series([np.array([1], dtype=np.int64)], dtype=object),
            "array_uint8": pd.Series([np.array([1], dtype=np.uint8)], dtype=object),
            "array_uint16": pd.Series([np.array([1], dtype=np.uint16)], dtype=object),
            "array_uint32": pd.Series([np.array([1], dtype=np.uint32)], dtype=object),
            "array_uint64": pd.Series([np.array([1], dtype=np.uint64)], dtype=object),
            "array_float32": pd.Series([np.array([1], dtype=np.float32)], dtype=object),
            "array_float64": pd.Series([np.array([1], dtype=np.float64)], dtype=object),
            "array_unicode": pd.Series([np.array(["Ö"], dtype=object)], dtype=object),
        }
    )

    return pd.concat([not_nested, nested_types], axis=1).reset_index(drop=True).sort_index(axis=1)
    # fmt: on


def get_dataframe_not_nested():
    return pd.DataFrame(
        {
            "bool": pd.Series([1], dtype=np.bool),
            "int8": pd.Series([1], dtype=np.int8),
            "int16": pd.Series([1], dtype=np.int16),
            "int32": pd.Series([1], dtype=np.int32),
            "int64": pd.Series([1], dtype=np.int64),
            "uint8": pd.Series([1], dtype=np.uint8),
            "uint16": pd.Series([1], dtype=np.uint16),
            "uint32": pd.Series([1], dtype=np.uint32),
            "uint64": pd.Series([1], dtype=np.uint64),
            "float32": pd.Series([1.0], dtype=np.float32),
            "float64": pd.Series([1.0], dtype=np.float64),
            "date": pd.Series([date(2018, 1, 1)], dtype=object),
            "datetime64": pd.Series(["2018-01-01"], dtype="datetime64[ns]"),
            "unicode": pd.Series(["Ö"], dtype=np.unicode),
            "null": pd.Series([None], dtype=object),
            # Adding a byte type with value as byte sequence which can not be encoded as UTF8
            "byte": pd.Series([gen_uuid_object().bytes], dtype=object),
        }
    ).sort_index(axis=1)


def get_scalar_dtype_strategy(exclude=None):
    """
    A `hypothesis` strategy yielding
    """
    possible_strategies = {
        "datetime": hyp_np.datetime64_dtypes(max_period="ms", min_period="ns"),
        "uint": hyp_np.unsigned_integer_dtypes(),
        "int": hyp_np.integer_dtypes(),
        "float": hyp_np.floating_dtypes(),
        "byte": hyp_np.byte_string_dtypes(),
        "unicode": hyp_np.unicode_string_dtypes(),
    }
    if exclude is None:
        exclude = {}
    elif not isinstance(exclude, list):
        exclude = [exclude]
    for ex in exclude:
        if ex in possible_strategies:
            del possible_strategies[ex]
        else:
            raise ValueError(
                "Strategy {} unknown. Possible values are {}".format(
                    ex, possible_strategies.keys()
                )
            )
    return hyp_st.one_of(*list(possible_strategies.values()))


def get_numpy_array_strategy(
    shape=10, exclude_dtypes=None, unique=False, sort=False, allow_nan=True
):
    # the text example generation has quite some overhead when called the first time.
    # we don't want this in our test sample generation since the HealthCheck of hypothesis
    # might be triggered.
    with catch_warnings():
        simplefilter("ignore", NonInteractiveExampleWarning)
        hyp_st.text().example()

    dtype_strategy = get_scalar_dtype_strategy(exclude_dtypes)
    array_strategy = hyp_np.arrays(dtype=dtype_strategy, shape=shape, unique=unique)

    if exclude_dtypes is None or "date" not in exclude_dtypes:
        date_start = hyp_st.lists(
            hyp_st.dates(
                min_value=datetime.date(1970, 1, 1), max_value=datetime.date(2100, 1, 1)
            ),
            min_size=shape,
            max_size=shape,
            unique=unique,
        )
        date_start = date_start.map(np.array)
        one_of_strategies = [array_strategy] + [date_start]
        array_strategy = hyp_st.one_of(one_of_strategies)

    def _restrict_datetime_ranges(arr):
        if np.issubdtype(arr.dtype, np.datetime64):
            return all(
                (arr < np.datetime64("2200-01-01"))
                & (arr > np.datetime64("1970-01-01"))
            )
        return True

    if exclude_dtypes is None or "datetime" not in exclude_dtypes:
        array_strategy = array_strategy.filter(_restrict_datetime_ranges)
    if not allow_nan:

        def _check_for_nan(arr):
            if np.issubdtype(arr.dtype, np.floating):
                return not any(np.isnan(arr))
            return True

        array_strategy = array_strategy.filter(_check_for_nan)
    if unique and allow_nan:

        def _maximum_single_nan(arr):
            if np.issubdtype(arr.dtype, np.floating):
                return sum(np.isnan(arr)) <= 1
            return True

        array_strategy = array_strategy.filter(_maximum_single_nan)

    if sort:
        array_strategy = array_strategy.map(np.sort)
    return array_strategy


@contextlib.contextmanager
def cm_frozen_time(time_to_freeze):
    """
    Context manager to monkeypatch kartothek.core._time.* to return
    a fixed datetime value `time_to_freeze`.
    """
    with mock.patch("kartothek.core._time.datetime_now") as mock_now, mock.patch(
        "kartothek.core._time.datetime_utcnow"
    ) as mock_utcnow:
        mock_now.return_value = time_to_freeze
        mock_utcnow.return_value = time_to_freeze
        yield
