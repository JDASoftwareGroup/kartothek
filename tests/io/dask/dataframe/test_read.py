import pickle
from functools import partial

import pandas as pd
import pytest

from kartothek.io.dask.dataframe import read_dataset_as_ddf
from kartothek.io.testing.read import *  # noqa
from kartothek.io_components.metapartition import SINGLE_TABLE


@pytest.fixture()
def output_type():
    return "table"


def _read_as_ddf(
    dataset_uuid,
    store,
    factory=None,
    categoricals=None,
    tables=None,
    dataset_has_index=False,
    **kwargs,
):
    table = tables or SINGLE_TABLE
    if categoricals:
        categoricals = categoricals[table]
    ddf = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        categoricals=categoricals,
        table=table,
        **kwargs,
    )
    if categoricals:
        assert ddf._meta.dtypes["P"] == pd.api.types.CategoricalDtype(
            categories=["__UNKNOWN_CATEGORIES__"], ordered=False
        )
        if dataset_has_index:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=[1, 2], ordered=False
            )
        else:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=["__UNKNOWN_CATEGORIES__"], ordered=False
            )

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    ddf = ddf.compute().reset_index(drop=True)

    def extract_dataframe(ix):
        df = ddf.iloc[[ix]].copy()
        for col in df.columns:
            if pd.api.types.is_categorical(df[col]):
                df[col] = df[col].cat.remove_unused_categories()
        return df.reset_index(drop=True)

    return [extract_dataframe(ix) for ix in ddf.index]


@pytest.fixture()
def bound_load_dataframes():
    return _read_as_ddf


def test_load_dataframe_categoricals_with_index(dataset_with_index_factory):
    func = partial(_read_as_ddf, dataset_has_index=True)
    test_read_dataset_as_dataframes(  # noqa: F405
        dataset_factory=dataset_with_index_factory,
        dataset=dataset_with_index_factory,
        store_session_factory=dataset_with_index_factory.store_factory,
        use_dataset_factory=True,
        bound_load_dataframes=func,
        use_categoricals=True,
        output_type="table",
        label_filter=None,
        dates_as_object=False,
    )
