# -*- coding: utf-8 -*-


import warnings
from typing import Callable, Dict, List, Optional, TypeVar, Union

import pandas as pd
from dask import delayed

from kartothek.core.factory import DatasetFactory
from kartothek.io_components.metapartition import MetaPartition

CATEGORICAL_EFFICIENCY_WARN_LIMIT = 100000

_ID_VALUE = TypeVar("_ID_VALUE")


def _identity() -> Callable[[_ID_VALUE], _ID_VALUE]:
    def _id(x):
        return x

    return _id


def _get_data(
    mp: MetaPartition, table: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Task to avoid serialization of lambdas
    """
    if table:
        return mp.data[table]
    else:
        return mp.data


def _cast_categorical_to_index_cat(
    df: pd.DataFrame, categories: Dict[str, str]
) -> pd.DataFrame:
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


def _construct_categorical(
    column: str, dataset_metadata_factory: DatasetFactory
) -> pd.CategoricalDtype:
    dataset_metadata = dataset_metadata_factory.load_index(column)
    values = dataset_metadata.indices[column].index_dct.keys()
    if len(values) > CATEGORICAL_EFFICIENCY_WARN_LIMIT:
        warnings.warn(
            f"Column {column} has {len(values)} distinct values, reading as categorical may increase memory consumption."
        )
    return pd.api.types.CategoricalDtype(values, ordered=False)


_CategoricalDtypeDict = Dict[str, Dict[str, pd.CategoricalDtype]]


def _maybe_get_categoricals_from_index(
    dataset_metadata_factory: DatasetFactory,
    categoricals: Optional[Dict[str, List[str]]],
) -> _CategoricalDtypeDict:
    """
    In case a categorical is requested for a column we have an index on,
    construct the categorical from the index
    """
    categoricals_from_index: _CategoricalDtypeDict = {}
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
