import pytest

from kartothek.io.eager import delete_dataset, garbage_collect_dataset
from kartothek.io.testing.delete import *  # noqa
from kartothek.io.testing.gc import *  # noqa: F4


def _delete_store_factory(dataset_uuid, store_factory):
    delete_dataset(dataset_uuid, store_factory)


def _delete_store(dataset_uuid, store_factory):
    delete_dataset(dataset_uuid, store_factory())


@pytest.fixture(params=["factory", "store-factory"])
def bound_delete_dataset(request):
    if request.param == "factory":
        return _delete_store_factory
    else:
        return _delete_store


@pytest.fixture()
def garbage_collect_callable():
    return garbage_collect_dataset
