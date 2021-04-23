"""
Global naming constants for datasets
"""


# FIXME: move this constant somewhere else. Cannot import from its declaration due to cyclic imports
SINGLE_TABLE = "table"
DEFAULT_METADATA_VERSION = 4
MIN_METADATA_VERSION = 4
MAX_METADATA_VERSION = 4

DEFAULT_METADATA_STORAGE_FORMAT = "json"

# Format suffixes
METADATA_FORMAT_JSON = ".json"
METADATA_FORMAT_MSGPACK = ".msgpack.zstd"

# Metadata suffixes
METADATA_BASE_SUFFIX = ".by-dataset-metadata"

# Object suffixes
PARQUET_FILE_SUFFIX = ".parquet"
EXTERNAL_INDEX_SUFFIX = f".by-dataset-index{PARQUET_FILE_SUFFIX}"

METADATA_VERSION_KEY = "dataset_metadata_version"
UUID_KEY = "dataset_uuid"

# Files/BLOB with special meaning

TABLE_METADATA_FILE = "_common_metadata"


def metadata_key_from_uuid(uuid, format="json"):
    if format == "json":
        return uuid + METADATA_BASE_SUFFIX + METADATA_FORMAT_JSON
    elif format == "msgpack":
        return uuid + METADATA_BASE_SUFFIX + METADATA_FORMAT_MSGPACK


def get_partition_file_prefix(
    dataset_uuid, partition_label, table, metadata_version=DEFAULT_METADATA_VERSION
):
    if metadata_version == 4:
        file_prefix = "{dataset_uuid}/{table}/{partition_label}".format(
            dataset_uuid=dataset_uuid, table=table, partition_label=partition_label
        )
    else:
        raise NotImplementedError(f"Metadata version {metadata_version} not supported")
    return file_prefix
