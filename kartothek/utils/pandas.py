"""
Pandas performance helpers.
"""
from __future__ import absolute_import

from collections import OrderedDict

import numpy as np
import pandas as pd

__all__ = (
    "aggregate_to_lists",
    "concat_dataframes",
    "drop_sorted_duplicates_keep_last",
    "is_dataframe_sorted",
    "mask_sorted_duplicates_keep_last",
    "merge_dataframes_robust",
    "sort_dataframe",
)


def concat_dataframes(dfs, default=None):
    """
    Concatenate given DataFrames.

    For non-empty iterables, this is roughly equivalent to::

        pd.concat(dfs, ignore_index=True, sort=False)

    except that the resulting index is undefined.

    .. important::

        If ``dfs`` is a list, it gets emptied during the process.


    .. warning::

        This requires all DataFrames to have the very same set of columns!

    Parameters
    ----------
    dfs: Iterable[pandas.DataFrame]
        Iterable of DataFrames w/ identical columns.
    default: Optional[pandas.DataFrame]
        Optional default if iterable is empty.

    Returns
    -------
    df: pandas.DataFrame
        Concatenated DataFrame or default value.

    Raises
    ------
    ValueError
        If iterable is empty but no default was provided.
    """
    # collect potential iterators
    if not isinstance(dfs, list):
        dfs = list(dfs)

    if len(dfs) == 0:
        if default is not None:
            res = default
        else:
            raise ValueError("Cannot concatenate 0 dataframes.")
    elif len(dfs) == 1:
        # that's faster than pd.concat w/ a single DF
        res = dfs[0]
    else:
        # pd.concat seems to hold the data in memory 3 times (not twice as you might expect it from naive copying the
        # input blocks into the output DF). This is very unfortunate especially for larger queries. This column-based
        # approach effectively reduces the maximum memory consumption and to our knowledge is not measuable slower.
        colset = set(dfs[0].columns)
        if not all(colset == set(df.columns) for df in dfs):
            raise ValueError("Not all DataFrames have the same set of columns!")

        res = pd.DataFrame(index=pd.RangeIndex(sum(len(df) for df in dfs)))
        for col in dfs[0].columns:
            res[col] = pd.concat(
                [df[col] for df in dfs], ignore_index=True, sort=False, copy=False
            )

    # ensure list (which is still referenced in parent scope) gets emptied
    del dfs[:]

    return res


def is_dataframe_sorted(df, columns):
    """
    Check that the given DataFrame is sorted as specified.

    This is more efficient than sorting the DataFrame.

    An empty DataFrame (no rows) is considered to be sorted.

    .. warning::

        This function does NOT handle NULL values correctly!

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to check.
    colums: Iterable[str]
        Column that the DataFrame should be sorted by.

    Returns
    -------
    sorted: bool
        ``True`` if DataFrame is sorted, ``False`` otherwise.

    Raises
    ------
    ValueError: If ``columns`` is empty.
    KeyError: If specified columns in ``by`` is missing.
    """
    columns = list(columns)

    if len(columns) == 0:
        raise ValueError("`columns` must contain at least 1 column")

    state = None

    for col in columns[::-1]:
        data = df[col].values
        if isinstance(data, pd.Categorical):
            data = np.asarray(data)
        data0 = data[:-1]
        data1 = data[1:]

        with np.errstate(invalid="ignore"):
            comp_le = data0 < data1
            comp_eq = data0 == data1

        if state is None:
            # last column
            state = comp_le | comp_eq
        else:
            state = comp_le | (comp_eq & state)

    return state.all()


def sort_dataframe(df, columns):
    """
    Sort DataFrame by columns.

    This is roughly equivalent to::

        df.sort_values(columns).reset_index(drop=True)

    .. warning::

        This function does NOT handle NULL values correctly!

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to sort.
    columns: Iterable[str]
        Columns to sort by.

    Returns
    -------
    df: pandas.DataFrame
        Sorted DataFrame w/ reseted index.
    """
    columns = list(columns)
    if is_dataframe_sorted(df, columns):
        return df
    data = [df[col].values for col in columns[::-1]]
    df = df.iloc[np.lexsort(data)]
    # reset inplace to reduce the memory usage
    df.reset_index(drop=True, inplace=True)
    return df


def mask_sorted_duplicates_keep_last(df, columns):
    """
    Mask duplicates on sorted data, keep last occurance as unique entry.

    Roughly equivalent to::

        df.duplicated(subset=columns, keep='last').values

    .. warning:
        NULL-values are not supported!

    .. warning:
        The behavior on unsorted data is undefined!

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame in question.
    columns: Iterable[str]
        Column-subset for duplicate-check (remaining columns are ignored).

    Returns
    -------
    mask: numpy.ndarray
        1-dimensional boolean array, marking duplicates w/ ``True``
    """
    columns = list(columns)
    rows = len(df)
    mask = np.zeros(rows, dtype=bool)

    if (rows > 1) and columns:
        sub = np.ones(rows - 1, dtype=bool)
        for col in columns:
            data = df[col].values
            sub &= data[:-1] == data[1:]
        mask[:-1] = sub

    return mask


def drop_sorted_duplicates_keep_last(df, columns):
    """
    Drop duplicates on sorted data, keep last occurance as unique entry.

    Roughly equivalent to::

        df.drop_duplicates(subset=columns, keep='last')

    .. warning:
        NULL-values are not supported!

    .. warning:
        The behavior on unsorted data is undefined!

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame in question.
    columns: Iterable[str]
        Column-subset for duplicate-check (remaining columns are ignored).

    Returns
    -------
    df: pandas.DataFrame
        DataFrame w/o duplicates.
    """
    columns = list(columns)
    dup_mask = mask_sorted_duplicates_keep_last(df, columns)
    if dup_mask.any():
        # pandas is just slow, so try to avoid the indexing call
        return df.iloc[~dup_mask]
    else:
        return df


def aggregate_to_lists(df, by, data_col):
    """
    Do a group-by and collect the results as python lists.

    Roughly equivalent to::

        df = df.groupby(
            by=by,
            as_index=False,
        )[data_col].agg(lambda series: list(series.values))

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe.
    by: Iterable[str]
        Group-by columns, might be empty.
    data_col: str
        Column with values to be collected.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame w/ operation applied.
    """
    by = list(by)

    if df.empty:
        return df

    if not by:
        return pd.DataFrame({data_col: pd.Series([list(df[data_col].values)])})

    # sort the DataFrame by `by`-values, so that rows of every group-by group are consecutive
    df = sort_dataframe(df, by)

    # collect the following data for every group:
    # - by-values
    # - list of values in `data_col`
    result_idx_data = [[] for _ in by]
    result_labels = []

    # remember index (aka values in `by`) and list of data values for current group
    group_idx = None  # Tuple[Any, ...]
    group_values = None  # List[Any]

    def _store_group():
        """
        Store current group from `group_idx` and `group_values` intro result lists.
        """
        if group_idx is None:
            # no group exists yet
            return

        for result_idx_part, idx_part in zip(result_idx_data, group_idx):
            result_idx_part.append(idx_part)
        result_labels.append(group_values)

    # create iterator over row-tuples, where every tuple contains values of all by-columns
    iterator_idx = zip(*(df[col].values for col in by))

    # iterate over all rows in DataFrame and collect groups
    for idx, label in zip(iterator_idx, df[data_col].values):
        if (group_idx is None) or (idx != group_idx):
            _store_group()
            group_idx = idx
            group_values = [label]
        else:
            group_values.append(label)

    # store last group
    _store_group()

    # create result DataFrame out of lists
    data = OrderedDict(zip(by, result_idx_data))
    data[data_col] = result_labels
    return pd.DataFrame(data)


def merge_dataframes_robust(df1, df2, how):
    """
    Merge two given DataFrames but also work if there are no columns to join on.

    If now shared column between the given DataFrames is found, then the join will be performaned on a single, constant
    column.

    Parameters
    ----------
    df1: pd.Dataframe
        Left DataFrame.
    df2: pd.Dataframe
        Right DataFrame.
    how: str
        How to join the frames.

    Returns
    -------
    df_joined: pd.DataFrame
        Joined DataFrame.
    """
    dummy_column = "__ktk_cube_join_dummy"

    columns2 = set(df2.columns)
    joined_columns = [c for c in df1.columns if c in columns2]

    if len(joined_columns) == 0:
        df1 = df1.copy()
        df2 = df2.copy()
        df1[dummy_column] = 1
        df2[dummy_column] = 1
        joined_columns = [dummy_column]

    df_out = df1.merge(df2, on=joined_columns, how=how, sort=False)

    df_out.drop(columns=dummy_column, inplace=True, errors="ignore")
    return df_out
