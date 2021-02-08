# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from kartothek.io.iter import update_dataset_from_dataframes__iter
from kartothek.io.testing.update import *  # noqa


@pytest.fixture()
def bound_update_dataset():
    return _update_dataset


def _update_dataset(df_list, *args, **kwargs):
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    df_generator = (x for x in df_list)
    return update_dataset_from_dataframes__iter(df_generator, *args, **kwargs)
