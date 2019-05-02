# -*- coding: utf-8 -*-

import pytest

from kartothek.io.iter import store_dataframes_as_dataset__iter
from kartothek.io.testing.write import *  # noqa


def _store_dataframes(df_list, *args, **kwargs):
    df_generator = (x for x in df_list)
    return store_dataframes_as_dataset__iter(df_generator, *args, **kwargs)


@pytest.fixture()
def bound_store_dataframes():
    return _store_dataframes
