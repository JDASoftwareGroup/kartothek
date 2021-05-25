# -*- coding: utf-8 -*-


import warnings
from functools import partial

import pandas as pd
from dask import delayed

try:
    from cytoolz import map
except ImportError:
    pass

CATEGORICAL_EFFICIENCY_WARN_LIMIT = 100000


def _identity():
    def _id(x):
        return x

    return _id


def _get_data(mp, table=None):
    """
    Task to avoid serialization of lambdas
    """
    if table:
        return mp.data[table]
    else:
        return mp.data


def _cast_categorical_to_index_cat(df, categories):
    try:
        return df.astype(categories)
    except ValueError as verr:
        # Should be fixed by pandas>=0.24.0
        if "buffer source array is read-only" in str(verr):
            new_cols = {}
            for cat in categories:
                new_cols[cat] = df[cat].astype(df[cat].dtype.categories.dtype)
                new_cols[cat] = new_cols[cat].astype(categories[cat])
            return df.assign(**new_cols)
        raise


def _construct_categorical(column, dataset_metadata_factory):
    dataset_metadata = dataset_metadata_factory.load_index(column)
    values = dataset_metadata.indices[column].index_dct.keys()
    if len(values) > CATEGORICAL_EFFICIENCY_WARN_LIMIT:
        warnings.warn(
            "Column {} has {} distinct values, reading as categorical may increase memory consumption.",
            column,
            len(values),
        )
    return pd.api.types.CategoricalDtype(values, ordered=False)


def _maybe_get_categoricals_from_index(dataset_metadata_factory, categoricals):
    """
    In case a categorical is requested for a column we have an index on,
    construct the categorical from the index
    """
    categoricals_from_index = {}
    if categoricals:
        for table, table_cat in categoricals.items():
            if not table_cat:
                continue
            categoricals_from_index[table] = {}
            for cat in table_cat:
                if cat in dataset_metadata_factory.indices:
                    cat_dtype = _construct_categorical(cat, dataset_metadata_factory)
                    categoricals_from_index[table][cat] = cat_dtype
    return categoricals_from_index


def map_delayed(func, mps, **kwargs):
    func = partial(func, **kwargs)
    delayed_func = delayed(func)
    return map(delayed_func, mps)
