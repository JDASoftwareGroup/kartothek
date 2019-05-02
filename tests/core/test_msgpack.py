# -*- coding: utf-8 -*-


from kartothek.core._zmsgpack import packb, unpackb


def test_msgpack():
    dct = {"a": 1, "b": {"c": "ÖaŒ"}}
    assert dct == unpackb(packb(dct))


def test_msgpack_storage(store):
    dct = {"a": 1, "b": {"c": "ÖaŒ"}}
    key = "test"
    store.put(key, packb(dct))
    value = store.get(key)
    assert dct == unpackb(value)
