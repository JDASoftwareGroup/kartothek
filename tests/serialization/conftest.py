import io

import numpy as np
import pytest

from kartothek.serialization.testing import BINARY_COLUMNS, get_dataframe_not_nested


class FakeStore:
    def __init__(self):
        self._brain = {}

    def put(self, key, value):
        self._brain[key] = value

    def get(self, key):
        return self._brain[key]

    def open(self, key):
        return io.BytesIO(self._brain[key])


@pytest.fixture
def store(tmpdir):
    return FakeStore()


@pytest.fixture(params=BINARY_COLUMNS)
def binary_value(request):
    return request.param


@pytest.fixture
def dataframe_not_nested():
    return get_dataframe_not_nested(10)


@pytest.fixture(autouse=True)
def numpy_errstate():
    with np.errstate(all="raise"):
        yield
