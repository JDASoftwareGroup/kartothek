import pkg_resources

from ._csv import CsvSerializer
from ._generic import (
    DataFrameSerializer,
    filter_array_like,
    filter_df,
    filter_df_from_predicates,
)
from ._parquet import ParquetSerializer

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:  # noqa
    __version__ = "unknown"


DataFrameSerializer.register_serializer(".csv.gz", CsvSerializer)
DataFrameSerializer.register_serializer(".csv", CsvSerializer)
DataFrameSerializer.register_serializer(".parquet", ParquetSerializer)


def default_serializer():
    return ParquetSerializer()


__all__ = [
    "DataFrameSerializer",
    "CsvSerializer",
    "ParquetSerializer",
    "default_serializer",
    "filter_df",
    "filter_array_like",
    "filter_df_from_predicates",
]
