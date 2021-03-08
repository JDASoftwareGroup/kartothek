"""
Workarounds for limitations of the simplekv API.
"""
import logging
import re
import time
from urllib.parse import quote

from simplekv.contrib import VALID_KEY_RE_EXTENDED

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.core.dataset import DatasetMetadataBuilder

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


def _copy_azure_bbs(keys, src_store, tgt_store):
    src_container = src_store.container
    tgt_container = tgt_store.container
    src_bbs = src_store.block_blob_service
    tgt_bbs = tgt_store.block_blob_service

    cprops = {}
    for k in keys:
        source_md5 = _azure_bbs_content_md5(
            src_bbs, src_container, k, accept_missing=False
        )

        if source_md5 is None:
            _logger.debug("Missing hash for {}".format(k))
        else:
            tgt_md5 = _azure_bbs_content_md5(
                tgt_bbs, tgt_container, k, accept_missing=True
            )

            if source_md5 == tgt_md5:
                _logger.debug("Omitting copy to {} (checksum match)".format(k))
                continue

        copy_source = src_bbs.make_blob_url(
            src_container, quote(k), sas_token=src_bbs.sas_token
        )
        cprops[k] = tgt_bbs.copy_blob(tgt_container, k, copy_source)

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


def _copy_azure_cc(keys, src_store, tgt_store):
    src_cc = src_store.blob_container_client
    tgt_cc = tgt_store.blob_container_client

    copy_ids = {}
    for k in keys:
        source_md5 = _azure_cc_content_md5(src_cc, k, accept_missing=False)

        if source_md5 is None:
            _logger.debug("Missing hash for {}".format(k))
        else:
            tgt_md5 = _azure_cc_content_md5(tgt_cc, k, accept_missing=True)

            if source_md5 == tgt_md5:
                _logger.debug("Omitting copy to {} (checksum match)".format(k))
                continue

        copy_source = src_cc.get_blob_client(k).url
        copy_ids[k] = tgt_cc.get_blob_client(k).start_copy_from_url(copy_source)[
            "copy_id"
        ]

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
    src_store: simplekv.KeyValueStore
        Source KV store
    tgt_store: simplekv.KeyValueStore
        Target KV store
    """
    for src_key, tgt_key in key_mappings.items():
        match_json = RX_JSON_META.match(src_key)
        if match_json:
            ds_name = match_json.groups()[0]
            item = mapped_metadata.get(ds_name, src_store.get(src_key))
        else:
            item = src_store.get(src_key)
        tgt_store.put(tgt_key, item)


def _transform_key(source_key, datasets_to_rename):
    """
    Transforms a source store key to a target store key. The transformation
    may contain renamings.

    Parameters
    ----------
    source_key: str
        Key inside the source store
    datasets_to_rename: Dict[str, str]
        Dictionary with datasets to rename, format {old_name: new:name}. Can be an
        empty dict if renaming is not planned
    """
    match_key = RX_TRANSFORM_KEY.match(source_key)
    if match_key:
        source_cube, source_dataset, source_path = match_key.groups()
        tgt_dataset = datasets_to_rename.get(source_dataset, source_dataset)
        return f"{source_cube}++{tgt_dataset}{source_path}"
    else:
        raise ValueError(f"Key {source_key} cannot be parsed!")


def _transform_metadata(datasets_to_rename, cube, src_store):
    result = {}
    for src_ds in datasets_to_rename.keys():
        src_metadata = discover_datasets_unchecked(
            uuid_prefix=cube.uuid_prefix,
            store=src_store,
            filter_ktk_cube_dataset_ids=[src_ds],
        )
        result[src_ds] = (
            DatasetMetadataBuilder.from_dataset(src_metadata[src_ds])
            .modify_dataset_name(datasets_to_rename[src_ds])
            .to_json()[1]
        )
    return result


def copy_keys(keys, src_store, tgt_store, datasets_to_rename=None, src_cube=None):
    """
    Copy keys from one store the another.

    Parameters
    ----------
    keys: Iterable[str]
        Keys to copy.
    src_store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Source KV store.
    tgt_store: Union[simplekv.KeyValueStore, Callable[[], simplekv.KeyValueStore]]
        Target KV store.
    datasets_to_rename: Optional[Dict[str, str]]
        Optional dict of {old dataset name:  new dataset name} entries. If specified,
        the corresponding datasets will be renamed accordingly.
    """
    if callable(src_store):
        src_store = src_store()
    if callable(tgt_store):
        tgt_store = tgt_store()

    # create dict {old key: new key} for all keys, where old and new may be empty
    # if no renaming is used
    mapped_keys = {}
    keys = sorted(keys)
    for k in keys:
        if (k is None) or (not VALID_KEY_RE_EXTENDED.match(k)) or (k == "/"):
            raise ValueError("Illegal key: {}".format(k))
        mapped_keys[k] = _transform_key(k, datasets_to_rename or {})

    if datasets_to_rename:
        mapped_metadata = _transform_metadata(datasets_to_rename, src_cube, src_store)
    else:
        mapped_metadata = {}
    if _has_azure_bbs(src_store) and _has_azure_bbs(tgt_store):
        _logger.debug(
            "Azure stores based on BlockBlobStorage class detected, use fast-path."
        )
        _copy_azure_bbs(keys, src_store, tgt_store)
    elif _has_azure_cc(src_store) and _has_azure_cc(tgt_store):
        _logger.debug(
            "Azure stores based on ContainerClient class detected, use fast-path."
        )
        _copy_azure_cc(keys, src_store, tgt_store)
    else:
        _logger.debug("Use naive slow-path.")
        _copy_naive(mapped_keys, src_store, tgt_store, mapped_metadata=mapped_metadata)
