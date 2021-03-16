"""
Workarounds for limitations of the simplekv API.
"""
import logging
import re
import time
from urllib.parse import quote

from simplekv.contrib import VALID_KEY_RE_EXTENDED

try:
    # azure-storage-blob < 12
    from azure.storage.blob import BlockBlobService as _BlockBlobService
    from azure.common import (
        AzureMissingResourceHttpError as _AzureMissingResourceHttpError,
    )
except ImportError:

    class _BlockBlobService:  # type: ignore
        """
        Dummy class.
        """

    class _AzureMissingResourceHttpError:  # type: ignore
        """
        Dummy class.
        """


try:
    # azure-storage-blob >= 12
    from azure.storage.blob import ContainerClient as _ContainerClient
    from azure.core.exceptions import ResourceNotFoundError as _ResourceNotFoundError
except ImportError:

    class _ContainerClient:  # type: ignore
        """
        Dummy class.
        """

    class _ResourceNotFoundError:  # type: ignore
        """
        Dummy class.
        """


__all__ = ("copy_keys",)


_logger = logging.getLogger(__name__)

RX_TRANSFORM_KEY = re.compile(r"(\w*)\+\+(\w*)(.*)")
RX_JSON_META = re.compile(r"\w*\+\+(\w*)\.by-dataset-metadata\.json")

# Specialized implementation for azure-storage-blob < 12, using BlockBlobService (`bbs`):


def _has_azure_bbs(store):
    try:
        # store decorators will forward getattr calls
        return isinstance(store.block_blob_service, _BlockBlobService)
    except AttributeError:
        return False


def _azure_bbs_content_md5(block_blob_service, container, key, accept_missing=False):
    try:
        return block_blob_service.get_blob_properties(
            container, key
        ).properties.content_settings.content_md5
    except _AzureMissingResourceHttpError:
        if accept_missing:
            return None
        else:
            raise KeyError(key)


def _copy_azure_bbs(key_mappings, src_store, tgt_store, mapped_metadata=None):
    src_container = src_store.container
    tgt_container = tgt_store.container
    src_bbs = src_store.block_blob_service
    tgt_bbs = tgt_store.block_blob_service

    # If metadata is to be modified while copying: put in manually in the target store
    if mapped_metadata:
        for src_key in mapped_metadata:
            tgt_key = key_mappings[src_key]
            tgt_store.put(tgt_key, mapped_metadata[src_key])

    cprops = {}
    for src_key, tgt_key in key_mappings.items():
        # skip modified metadata which was already copied manually
        if mapped_metadata and (src_key in mapped_metadata):
            continue

        source_md5 = _azure_bbs_content_md5(
            src_bbs, src_container, src_key, accept_missing=False
        )

        if source_md5 is None:
            _logger.debug("Missing hash for {}".format(src_key))
        else:
            tgt_md5 = _azure_bbs_content_md5(
                tgt_bbs, tgt_container, tgt_key, accept_missing=True
            )

            if source_md5 == tgt_md5:
                _logger.debug(
                    "Omitting copy from {} to {} (checksum match)".format(
                        src_key, tgt_key
                    )
                )
                continue

        copy_source = src_bbs.make_blob_url(
            src_container, quote(src_key), sas_token=src_bbs.sas_token
        )
        cprops[tgt_key] = tgt_bbs.copy_blob(tgt_container, tgt_key, copy_source)

    for k, cprop in cprops.items():
        while True:
            blob = tgt_bbs.get_blob_properties(tgt_container, k)
            cprop_current = blob.properties.copy
            assert cprop.id == cprop_current.id, "Concurrent copy to {}".format(k)
            if cprop_current.status == "pending":
                _logger.debug("Waiting for pending copy to {}...".format(k))
                time.sleep(0.1)
                continue
            elif cprop_current.status == "success":
                _logger.debug("Copy to {} completed".format(k))
                break  # break from while, continue in for-loop
            else:
                raise RuntimeError(
                    "Error while copying: status is {}: {}".format(
                        cprop_current.status, cprop_current.status_description
                    )
                )


# Specialized implementation for azure-storage-blob >= 12, using ContainerClient (`cc`):
def _has_azure_cc(store):
    try:
        # store decorators will forward getattr calls
        return isinstance(store.blob_container_client, _ContainerClient)
    except AttributeError:
        return False


def _azure_cc_content_md5(cc, key, accept_missing=False):
    try:
        bc = cc.get_blob_client(key)
        return bc.get_blob_properties().content_settings.content_md5
    except _ResourceNotFoundError:
        if accept_missing:
            return None
        else:
            raise KeyError(key)


def _copy_azure_cc(key_mappings, src_store, tgt_store, mapped_metadata=None):
    """
    Copies a list of items from one Azure store to another.

    Parameters
    ----------
    key_mappings: Dict[str, str]
        Mapping of source key names to target key names. May be equal if a key will
        not be renamed.
    src_store: KeyValueStore
        Source KV store
    tgt_store: KeyValueStore
        Target KV store
    mapped_metadata: Optional[Dict[str, bytes]]
        Mapping containing {key: modified metadata} entries with metadata which is
        to be changed during copying
    """
    src_cc = src_store.blob_container_client
    tgt_cc = tgt_store.blob_container_client

    # If metadata is to be modified while copying: put in manually in the target store
    if mapped_metadata:
        for src_key in mapped_metadata:
            tgt_key = key_mappings[src_key]
            tgt_store.put(tgt_key, mapped_metadata[src_key])

    copy_ids = {}
    for src_key, tgt_key in key_mappings.items():
        # skip modified metadata which was already copied manually
        if mapped_metadata and (src_key in mapped_metadata):
            continue
        source_md5 = _azure_cc_content_md5(src_cc, src_key, accept_missing=False)

        if source_md5 is None:
            _logger.debug("Missing hash for {}".format(src_key))
        else:
            tgt_md5 = _azure_cc_content_md5(tgt_cc, tgt_key, accept_missing=True)

            if source_md5 == tgt_md5:
                _logger.debug(
                    "Omitting copy from {} to {} (checksum match)".format(
                        src_key, tgt_key
                    )
                )
                continue

        copy_source = src_cc.get_blob_client(src_key).url
        copy_ids[tgt_key] = tgt_cc.get_blob_client(tgt_key).start_copy_from_url(
            copy_source
        )["copy_id"]

    for k, copy_id in copy_ids.items():
        while True:
            cprop_current = tgt_cc.get_blob_client(k).get_blob_properties().copy
            assert copy_id == cprop_current.id, "Concurrent copy to {}".format(k)
            if cprop_current.status == "pending":
                _logger.debug("Waiting for pending copy to {}...".format(k))
                time.sleep(0.1)
                continue
            elif cprop_current.status == "success":
                _logger.debug("Copy to {} completed".format(k))
                break  # break from while, continue in for-loop
            else:
                raise RuntimeError(
                    "Error while copying: status is {}: {}".format(
                        cprop_current.status, cprop_current.status_description
                    )
                )


def _copy_naive(key_mappings, src_store, tgt_store, mapped_metadata=None):
    """
    Copies a list of items from one KV store to another.

    Parameters
    ----------
    key_mappings: Dict[str, str]
        Mapping of source key names to target key names. May be equal if a key will
        not be renamed.
    src_store: KeyValueStore
        Source KV store
    tgt_store: KeyValueStore
        Target KV store
    mapped_metadata: Dict[str, bytes]
        Mapping containing {key: modified metadata} values to be changed
    """
    for src_key, tgt_key in key_mappings.items():
        if mapped_metadata & (src_key in mapped_metadata):
            item = mapped_metadata.get(src_key)
        else:
            item = src_store.get(src_key)
        tgt_store.put(tgt_key, item)


def copy_keys(keys, src_store, tgt_store, md_transformed=None):
    """
    Copy keys from one store the another.

    Parameters
    ----------
    keys: Union[Iterable[str], Dict[str, str]]
        Either Iterable of Keys to copy; or Dict with {old key: new key} mappings
        if keys are changed during copying
    src_store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Source KV store.
    tgt_store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Target KV store.
    md_transformed: Dict[str, str]

    """
    if callable(src_store):
        src_store = src_store()
    if callable(tgt_store):
        tgt_store = tgt_store()

    if isinstance(keys, dict):
        # If a dict of keys was specified, i.e. the keys are to be renamed while
        # copying: use this key mapping
        src_keys = sorted(keys.keys())
        key_mappings = keys
    else:
        # otherwise, create a identity mapping dict which does not change the key
        src_keys = sorted(keys)
        key_mappings = {src_key: src_key for src_key in src_keys}

    for k in src_keys:
        if (k is None) or (not VALID_KEY_RE_EXTENDED.match(k)) or (k == "/"):
            raise ValueError("Illegal key: {}".format(k))

    if _has_azure_bbs(src_store) and _has_azure_bbs(tgt_store):
        _logger.debug(
            "Azure stores based on BlockBlobStorage class detected, use fast-path."
        )
        _copy_azure_bbs(key_mappings, src_store, tgt_store)
    elif _has_azure_cc(src_store) and _has_azure_cc(tgt_store):
        _logger.debug(
            "Azure stores based on ContainerClient class detected, use fast-path."
        )
        _copy_azure_cc(key_mappings, src_store, tgt_store)
    else:
        _logger.debug("Use naive slow-path.")
        _copy_naive(key_mappings, src_store, tgt_store, mapped_metadata=md_transformed)
