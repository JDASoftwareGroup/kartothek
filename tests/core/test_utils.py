from functools import partial

import pytest
from simplekv import KeyValueStore
from simplekv.decorator import PrefixDecorator
from storefact import get_store_from_url

from kartothek.core.utils import ensure_store, lazy_store


class NoPickle:
    def __getstate__(self):
        raise RuntimeError("do NOT pickle this object!")


def mark_nopickle(obj):
    setattr(obj, "_nopickle", NoPickle())


def no_pickle_store(url):
    store = get_store_from_url(url)
    mark_nopickle(store)
    return store


def no_pickle_factory(url):

    return partial(no_pickle_store, url)


@pytest.fixture(params=["URL", "KeyValue", "Callable"])
def store_input_types(request, tmpdir):
    url = f"hfs://{tmpdir}"

    if request.param == "URL":
        return url
    elif request.param == "KeyValue":
        return get_store_from_url(url)
    elif request.param == "Callable":
        return no_pickle_factory(url)
    else:
        raise RuntimeError(f"Encountered unknown store type {type(request.param)}")


def test_ensure_store(store_input_types):
    store = ensure_store(store_input_types)
    assert isinstance(store, KeyValueStore)
    value = b"value"
    key = "key"
    store.put(key, value)
    assert value == store.get(key)

    assert store is ensure_store(store)


def test_ensure_store_fact(store_input_types):
    store_fact = lazy_store(store_input_types)
    assert callable(store_fact)
    store = store_fact()
    assert isinstance(store, KeyValueStore)
    value = b"value"
    key = "key"
    store.put(key, value)
    assert value == store.get(key)

    assert store_fact is lazy_store(store_fact)


def test_ensure_store_returns_same_store():
    store = get_store_from_url("memory://")
    assert ensure_store(lambda: store) is store


def test_lazy_store_returns_same_store():
    store = get_store_from_url("memory://")
    assert lazy_store(lambda: store)() is store


def test_lazy_store_accepts_decorated_store():
    store = get_store_from_url("memory://")
    pstore = PrefixDecorator("pre", store)
    assert lazy_store(pstore)() is pstore
