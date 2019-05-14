# -*- coding: utf-8 -*-


import warnings

import pandas as pd
from dask import delayed

CATEGORICAL_EFFICIENCY_WARN_LIMIT = 100000


def _construct_categorical(column, dataset_metadata_factory):
    dataset_metadata = dataset_metadata_factory.load_index(column)
    values = dataset_metadata.indices[column].index_dct.keys()
    if len(values) > CATEGORICAL_EFFICIENCY_WARN_LIMIT:
        warnings.warn(
            "Column {} has {} distinct values, reading as categorical may increase memory consumption.",
            column,
            len(values),
        )
    return pd.api.types.CategoricalDtype(values)


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


def map_delayed(mps, func, *args, **kwargs):
    return [delayed(func)(mp, *args, **kwargs) for mp in mps]
