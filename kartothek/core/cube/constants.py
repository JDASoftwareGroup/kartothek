"""
Common constants for Kartothek.
"""
from kartothek.serialization import ParquetSerializer

__all__ = (
    "KTK_CUBE_DF_SERIALIZER",
    "KTK_CUBE_METADATA_DIMENSION_COLUMNS",
    "KTK_CUBE_METADATA_KEY_IS_SEED",
    "KTK_CUBE_METADATA_STORAGE_FORMAT",
    "KTK_CUBE_METADATA_VERSION",
    "KTK_CUBE_UUID_SEPERATOR",
)


#
# !!!! WARNING !!!
#
# If you change any of these constants, this may break backwards compatibility.
# Also, always ensure to also adapt the docs (especially the format specification in the README).
#
# !!!!!!!!!!!!!!!!
#


#: DataFrame serializer that is be used to write data.
KTK_CUBE_DF_SERIALIZER = ParquetSerializer(compression="ZSTD")

#: Storage format for kartothek metadata that is be used by default.
KTK_CUBE_METADATA_STORAGE_FORMAT = "json"

#: Kartothek metadata version that ktk_cube is based on.
KTK_CUBE_METADATA_VERSION = 4

#: Metadata key that is used to mark seed datasets
KTK_CUBE_METADATA_KEY_IS_SEED = "klee_is_seed"

#: Metadata key to store dimension columns
KTK_CUBE_METADATA_DIMENSION_COLUMNS = "klee_dimension_columns"

#: Metadata key to store partition columns
KTK_CUBE_METADATA_PARTITION_COLUMNS = "klee_partition_columns"

#: Metadata key to store the timestamp column (can be null)
KTK_CUBE_METADATA_TIMESTAMP_COLUMN = "klee_timestamp_column"

#: Character sequence used to seperate cube and dataset UUID
KTK_CUBE_UUID_SEPERATOR = "++"
