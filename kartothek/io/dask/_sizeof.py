from dask.sizeof import sizeof as dask_sizeof


def _dct_sizeof(obj):
    return dask_sizeof(obj.__dict__)


def register_sizeof_ktk_classes():
    from kartothek.core.dataset import DatasetMetadata
    from kartothek.core.factory import DatasetFactory
    from kartothek.io_components.metapartition import MetaPartition
    from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
    from kartothek.core.partition import Partition
    from kartothek.core.common_metadata import SchemaWrapper

    dask_sizeof.register(DatasetMetadata, _dct_sizeof)
    dask_sizeof.register(DatasetFactory, _dct_sizeof)
    dask_sizeof.register(MetaPartition, _dct_sizeof)
    dask_sizeof.register(ExplicitSecondaryIndex, _dct_sizeof)
    dask_sizeof.register(PartitionIndex, _dct_sizeof)
    dask_sizeof.register(Partition, _dct_sizeof)
    dask_sizeof.register(SchemaWrapper, _dct_sizeof)
