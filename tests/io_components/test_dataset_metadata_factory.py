import pickle
from copy import copy
from functools import partial

import pytest

from kartothek.core.factory import DatasetFactory


class CountStore:
    def __init__(self, inner):
        self.inner = inner
        self.get_count = 0

    def get(self, key):
        self.get_count += 1
        return self.inner.get(key)

    def iter_keys(self, prefix):
        return iter(self)

    def __iter__(self):
        return self.inner.__iter__()


class CountFactory:
    def __init__(self, inner):
        self.inner = inner
        self.count = 0
        self.last = None

    def __call__(self):
        self.count += 1
        result = self.inner()
        self.last = result
        return result

    def __getstate__(self):
        state = copy(self.__dict__)
        state["last"] = None
        return state


def _create_count_store(store_factory):
    return CountStore(store_factory())


@pytest.fixture(scope="function")
def count_store(store_factory):
    return CountFactory(partial(_create_count_store, store_factory))


def test_store_init(count_store, dataset_function):
    factory = DatasetFactory(dataset_uuid="dataset_uuid", store_factory=count_store)
    assert count_store.count == 0

    store = factory.store
    assert hasattr(store, "get")

    assert count_store.count == 1
    assert count_store.last == store

    assert store.get_count == 0

    # second get should cache
    store = factory.store
    assert count_store.count == 1
    assert count_store.last == store


def test_uuid(count_store, dataset_function):
    factory = DatasetFactory(dataset_uuid="dataset_uuid", store_factory=count_store)
    assert factory.dataset_uuid == "dataset_uuid"


def test_get_metadata(count_store, dataset_function):
    factory = DatasetFactory(dataset_uuid="dataset_uuid", store_factory=count_store)
    store = factory.store
    assert store.get_count == 0

    metadata = factory.dataset_metadata
    assert hasattr(metadata, "metadata")

    initial_count = store.get_count

    # second get should cache
    metadata = factory.dataset_metadata
    assert store.get_count == initial_count


def test_pickle(count_store, dataset_function):
    factory = DatasetFactory(dataset_uuid="dataset_uuid", store_factory=count_store)
    assert factory._cache_store is None
    assert factory._cache_metadata is None

    factory.store
    factory.dataset_metadata
    assert factory._cache_store is not None
    assert factory._cache_metadata is not None

    factory2 = pickle.loads(pickle.dumps(factory, pickle.HIGHEST_PROTOCOL))
    assert factory2._cache_store is None
    assert factory2._cache_metadata is None
