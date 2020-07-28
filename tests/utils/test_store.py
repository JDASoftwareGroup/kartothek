# -*- coding: utf-8 -*-
import base64
import hashlib

import pytest
import storefact

from kartothek.utils.store import (
    _azure_bbs_content_md5,
    _azure_cc_content_md5,
    _has_azure_bbs,
    _has_azure_cc,
    copy_keys,
)

STORE_TYPES = ["azure", "fs", "memory"]

try:
    from azure.storage.blob import __version__ as asb_version

    asb_major_version = int(asb_version.split(".")[0])
except ImportError:
    asb_major_version = 0


def _gen_store(store_type, tmpdir, suffix, azure_store_cfg_factory):
    if store_type == "azure":
        cfg = azure_store_cfg_factory(suffix)
    elif store_type == "fs":
        cfg = {"type": "hfs", "path": tmpdir.join(suffix).strpath}
    elif store_type == "memory":
        cfg = {"type": "hmemory"}
    else:
        raise ValueError("Unknown store type: {}".format(store_type))

    store = storefact.get_store(**cfg)
    for k in store.keys():
        store.delete(k)

    yield store

    for k in store.keys():
        store.delete(k)

    # prevent ResourceWarning
    if hasattr(store, "block_blob_service"):
        store.block_blob_service.request_session.close()
    if hasattr(store, "blob_container_client"):
        store.blob_container_client.close()


@pytest.fixture(params=STORE_TYPES)
def store(request, tmpdir, azure_store_cfg_factory):
    for s in _gen_store(request.param, tmpdir, "store", azure_store_cfg_factory):
        yield s


@pytest.fixture(params=STORE_TYPES)
def store2(request, tmpdir, azure_store_cfg_factory):
    for s in _gen_store(request.param, tmpdir, "store2", azure_store_cfg_factory):
        yield s


@pytest.mark.skipif(
    asb_major_version != 2,
    reason=f"Test is specific for azure-storage-blob ~= 2, but detected version {asb_major_version}",
)
def test_azure_implementation(azure_store_cfg_factory):
    cfg = azure_store_cfg_factory("ts")
    store = storefact.get_store(**cfg)
    assert _has_azure_bbs(store)
    content = b"foo"
    store.put("key0", content)
    assert (
        base64.b64decode(
            _azure_bbs_content_md5(store.block_blob_service, store.container, "key0")
        ).hex()
        == hashlib.md5(content).hexdigest()
    )


@pytest.mark.skipif(
    asb_major_version < 12,
    reason=f"Test is specific for azure-storage-blob >= 12, but detected version {asb_major_version}",
)
def test_azure12_implementation(azure_store_cfg_factory):
    cfg = azure_store_cfg_factory("ts")
    store = storefact.get_store(**cfg)
    assert _has_azure_cc(store)
    content = b"foobar"
    store.put("key0", content)
    assert (
        _azure_cc_content_md5(store.blob_container_client, "key0").hex()
        == hashlib.md5(content).hexdigest()
    )
    assert (
        _azure_cc_content_md5(store.blob_container_client, "key_does_not_exist", True)
        is None
    )


class TestCopy:
    def test_all(self, store, store2):
        store.put("ka", b"ka")
        store.put("kb", b"kb")

        copy_keys({"ka", "kb"}, store, store2)

        assert set(store2.keys()) == {"ka", "kb"}
        assert store2.get("ka") == b"ka"
        assert store2.get("kb") == b"kb"

    def test_part(self, store, store2):
        store.put("ka", b"ka")
        store.put("kb", b"kb")
        store.put("kc", b"kc")

        copy_keys({"ka", "kb"}, store, store2)

        assert set(store2.keys()) == {"ka", "kb"}
        assert store2.get("ka") == b"ka"
        assert store2.get("kb") == b"kb"

    def test_overwrite(self, store, store2):
        store.put("ka", b"ka")

        store2.put("ka", b"la")
        store2.put("kb", b"lb")

        copy_keys({"ka"}, store, store2)

        assert store2.get("ka") == b"ka"
        assert store2.get("kb") == b"lb"

    def test_same(self, store):
        store.put("ka", b"ka")
        store.put("kb", b"kb")

        copy_keys({"ka", "kb"}, store, store)

        assert set(store.keys()) == {"ka", "kb"}
        assert store.get("ka") == b"ka"
        assert store.get("kb") == b"kb"

    def test_missing(self, store, store2):
        store.put("ka", b"ka")

        with pytest.raises(KeyError) as exc:
            copy_keys({"ka", "kb"}, store, store2)
        assert str(exc.value) == "'kb'"

    def test_weird1(self, store, store2):
        k = "k / =+"
        store.put(k, b"foo")

        copy_keys({k}, store, store2)

        assert set(store2.keys()) == {k}
        assert store2.get(k) == b"foo"

    def test_weird2(self, store, store2):
        # regression found during early customer tests
        k = "cube_main++completeness/table/PARTITION_DATE=2018-06-29%2000%3A00%3A00/PARTITION_ID=366/KLEE_TS=2019-02-27%2012%3A59%3A37.904401/9be13a70302441b2aee5bed2af781be7.parquet"
        store.put(k, b"foo")

        copy_keys({k}, store, store2)

        assert set(store2.keys()) == {k}
        assert store2.get(k) == b"foo"

    def test_illegal(self, store, store2):
        with pytest.raises(ValueError) as exc:
            copy_keys({"kö"}, store, store2)
        assert str(exc.value) == "Illegal key: kö"
