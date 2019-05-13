# -*- coding: utf-8 -*-

import pytest

from kartothek.io.eager import garbage_collect_dataset
from kartothek.io.testing.gc import *  # noqa: F4


@pytest.fixture()
def garbage_collect_callable():
    return garbage_collect_dataset
