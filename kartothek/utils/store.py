"""
Workarounds for limitations of the minimalkv API.
"""
import logging
import time
from typing import Callable, Dict, Iterable, Optional, Union
from urllib.parse import quote

from minimalkv import KeyValueStore
from minimalkv.contrib import VALID_KEY_RE_EXTENDED

from kartothek.core.dataset import DatasetMetadata

try:
    # azure-storage-blob < 12
    from azure.common import (
        AzureMissingResourceHttpError as _AzureMissingResourceHttpError,
    )
    from azure.storage.blob import BlockBlobService as _BlockBlobService
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
    from azure.core.exceptions import ResourceNotFoundError as _ResourceNotFoundError
    from azure.storage.blob import ContainerClient as _ContainerClient
except ImportError:

    class _ContainerClient:  # type: ignore
        """
        Dummy class.
        """

    class _ResourceNotFoundError:  # type: ignore
        """
        Dummy class.
        """


__all__ = ("copy_keys", "copy_rename_keys")


_logger = logging.getLogger(__name__)


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


def _copy_azure_bbs(key_mappings, src_store, tgt_store):
    src_container = src_store.container
    tgt_container = tgt_store.container
    src_bbs = src_store.block_blob_service
    tgt_bbs = tgt_store.block_blob_service

    cprops = {}
    for src_key, tgt_key in key_mappings.items():
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


def _copy_azure_cc(key_mappings, src_store, tgt_store):
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
    """
    src_cc = src_store.blob_container_client
    tgt_cc = tgt_store.blob_container_client

    copy_ids = {}
    for src_key, tgt_key in key_mappings.items():
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


def _copy_naive(
    key_mappings: Dict[str, str],
    src_store: KeyValueStore,
    tgt_store: KeyValueStore,
    md_transformed: Optional[Dict[str, DatasetMetadata]] = None,
):
    """
    Copies a list of items from one KV store to another.
    Parameters
    ----------
    key_mappings: Dict[str, str]
        Mapping of source key names to target key names. May be equal if a key will
        not be renamed.
    src_store: minimalkv.KeyValueStore
        Source KV store–
    tgt_store: minimalkv.KeyValueStore
        Target KV store
    md_transformed: Dict[str, DatasetMetadata]
        Mapping containing {target dataset uuid: modified target metadata} values which will be written
        directly instead of being copied
    """
    for src_key, tgt_key in key_mappings.items():
        if (md_transformed is not None) and (tgt_key in md_transformed):
            item = md_transformed.get(tgt_key).to_json()  # type: ignore
        else:
            item = src_store.get(src_key)
        tgt_store.put(tgt_key, item)


def copy_rename_keys(
    key_mappings: Dict[str, str],
    src_store: KeyValueStore,
    tgt_store: KeyValueStore,
    md_transformed: Dict[str, DatasetMetadata],
):
    """
    Copy keys between to stores or within one store, and rename them.
    Parameters
    ----------
    key_mappings: Dict[str, str]
        Dict with {old key: new key} mappings to rename keys during copying
    src_store: minimalkv.KeyValueStore
        Source KV store.
    tgt_store: minimalkv.KeyValueStore
        Target KV store.
    md_transformed:
        Mapping of the new target dataset uuid to the new and potentially renamed metadata of the copied dataset.
    """
    for k in key_mappings.keys():
        if (k is None) or (not VALID_KEY_RE_EXTENDED.match(k)) or (k == "/"):
            raise ValueError("Illegal key: {}".format(k))
    _logger.debug("copy_rename_keys: Use naive slow-path.")
    _copy_naive(key_mappings, src_store, tgt_store, md_transformed)


def copy_keys(
    keys: Iterable[str],
    src_store: Union[KeyValueStore, Callable[[], KeyValueStore]],
    tgt_store: Union[KeyValueStore, Callable[[], KeyValueStore]],
):
    """
    Copy keys between two stores or within one store.

    Parameters
    ----------
    keys: Iterable[str]
        Set of keys to copy without renaming;
    src_store: Union[minimalkv.KeyValueStore, Callable[[], minimalkv.KeyValueStore]]
        Source KV store.
    tgt_store: Union[minimalkv.KeyValueStore, Callable[[], minimalkv.KeyValueStore]]
        Target KV store.
    """
    if callable(src_store):
        src_store = src_store()
    if callable(tgt_store):
        tgt_store = tgt_store()

    keys = sorted(keys)
    # create a identity mapping dict which does not change the key
    key_mappings = {src_key: src_key for src_key in keys}

    for k in keys:
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
        _copy_naive(key_mappings, src_store, tgt_store)
