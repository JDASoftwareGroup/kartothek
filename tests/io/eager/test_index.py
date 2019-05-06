# -*- coding: utf-8 -*-

import pytest

from kartothek.io.eager import build_dataset_indices
from kartothek.io.testing.index import *  # noqa: F4


@pytest.fixture()
def bound_build_dataset_indices():
    return build_dataset_indices
