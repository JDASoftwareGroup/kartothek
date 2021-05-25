from functools import partial

import pytest

from kartothek.io.iter import (
    read_dataset_as_dataframes__iterator,
    read_dataset_as_metapartitions__iterator,
)
from kartothek.io.testing.read import *  # noqa


@pytest.fixture(params=["dataframe", "metapartition"])
def output_type(request):
    return request.param


def _load_dataframes(output_type, *args, **kwargs):
    if output_type == "dataframe":
        func = read_dataset_as_dataframes__iterator
    elif output_type == "metapartition":
        func = read_dataset_as_metapartitions__iterator
    else:
        raise ValueError("Unknown output type {}".format(output_type))
    return list(func(*args, **kwargs))


@pytest.fixture()
def bound_load_dataframes(output_type):
    return partial(_load_dataframes, output_type)


def _load_metapartitions(*args, **kwargs):
    return list(read_dataset_as_metapartitions__iterator(*args, **kwargs))


@pytest.fixture()
def bound_load_metapartitions():
    return _load_metapartitions
