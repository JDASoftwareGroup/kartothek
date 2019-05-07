# -*- coding: utf-8 -*-

import pytest

from kartothek.io.eager import update_dataset_from_dataframes
from kartothek.io.testing.update import *  # noqa: F40


@pytest.fixture()
def bound_update_dataset():
    return update_dataset_from_dataframes
