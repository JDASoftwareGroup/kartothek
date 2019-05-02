# -*- coding: utf-8 -*-


import pandas as pd

from kartothek.core.testing import get_dataframe_alltypes


def test_get_dataframe_alltypes():
    df = get_dataframe_alltypes()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "byte" in df.columns
