from typing import Any, Dict, List, Optional

import attr
import dask
import dask.bag as db
import dask.dataframe as dd
import pandas as pd

from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.serialization import ParquetSerializer

from .common_metadata import SchemaWrapper
from .dataset import _validate_uuid
from .factory import DatasetFactory
from .utils import _check_callable


def _wrap_simple_validator(func):
    def _(self, attr, value):
        return func(value)

    return _


@attr.s
class Table:
    dataset_uuid = attr.ib(validator=_wrap_simple_validator(_validate_uuid), type=str)
    store_factory = attr.ib(validator=_wrap_simple_validator(_check_callable), type=Any)
    name = attr.ib(default=SINGLE_TABLE, type=str)

    def __attrs_post_init__(self):
        self._factory = DatasetFactory(self.dataset_uuid, self.store_factory)

    @staticmethod
    def from_factory(factory: DatasetFactory, name=SINGLE_TABLE) -> "Table":
        inst = Table(factory.dataset_uuid, factory.store_factory, name)
        inst._factory = factory
        return inst

    @property
    def schema(self) -> SchemaWrapper:
        return self._factory.table_meta[self.name]

    def read(self, **kwargs) -> "TableReader":
        return TableReader(self, **kwargs)

    def write(self, **kwargs) -> "TableWriter":
        return TableWriter(self, **kwargs)

    def __getattr__(self, name: str):
        # __getattr__ should only be called if the attribute cannot be found. if the
        # attribute is None, it still falls back to this call
        factory = getattr(self, "_factory")
        return getattr(factory, name)


@attr.s
class TableReader:
    table = attr.ib(type=Table)

    predicates = attr.ib(type=Any, default=None, kw_only=True)
    columns = attr.ib(type=Optional[List[str]], default=None, kw_only=True)
    dispatch_by = attr.ib(type=Optional[List[str]], default=None, kw_only=True)
    predicate_pushdown_to_io = attr.ib(type=Optional[bool], default=None, kw_only=True)
    categories = attr.ib(type=Optional[List[str]], default=None, kw_only=True)
    dates_as_object = attr.ib(type=Optional[bool], default=None, kw_only=True)

    def set_option(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def to_delayed(self) -> List[dask.delayed]:
        raise NotImplementedError

    def to_ddf(self) -> dd.DataFrame:
        raise NotImplementedError

    def to_bag(self) -> db.Bag:
        raise NotImplementedError

    def to_pandas(self) -> pd.DataFrame:
        from kartothek.io.eager import read_table

        return read_table(
            factory=self.table._factory,
            predicates=self.predicates,
            columns=self.columns,
            table=self.table.name,
            categoricals=self.categories,
            dates_as_object=self.dates_as_object,
        )


@attr.s
class TableWriter:

    table = attr.ib(type=Table)

    secondary_indices = attr.ib(type=Optional[List[str]], default=None, kw_only=True)
    partition_on = attr.ib(type=Optional[List[str]], default=None, kw_only=True)
    metadata = attr.ib(type=Optional[Dict], default=None, kw_only=True)
    metadata_version = attr.ib(type=Optional[str], default=None, kw_only=True)
    sort_partitions_by = attr.ib(type=Optional[str], default=None, kw_only=True)
    append_partitions = attr.ib(type=Optional[List], default=None, kw_only=True)
    delete_scope = attr.ib(type=Optional[List[Dict]], default=None, kw_only=True)
    allow_override = attr.ib(type=Optional[bool], default=False, kw_only=True)
    compression = attr.ib(type=Optional[str], default="zstd", kw_only=True)
    parquet_chunksize = attr.ib(type=Optional[int], default=None, kw_only=True)
    default_metadata_version = attr.ib(type=int, default=4, kw_only=True)

    def copy(self, **kwargs):
        return attr.evolve(self, **kwargs)

    @property
    def TYPED_TABLEWRITER(self):
        return {
            dd.DataFrame: DaskDataFrameTableWriter,
            db.Bag: DaskDataFrameTableWriter,
        }

    @property
    def df_serializer(self):
        return ParquetSerializer(
            chunk_size=self.parquet_chunksize, compression=self.compression
        )

    def commit(self, compute=True):
        raise NotImplementedError

    def _get_contructor_from_partitions(self, obj):
        if isinstance(obj, list) and obj:
            return self._get_contructor_from_partitions(obj[0])
        else:
            return self.TYPED_TABLEWRITER.get(type(obj), BaseTableWriter)

    def add_partitions(self, partitions=None, ddf=None) -> "TableWriter":
        writer_class = self._get_contructor_from_partitions(partitions)
        options = attr.asdict(self)
        del options["table"]
        instance = writer_class(table=self.table, **options)
        return instance.copy(append_partitions=partitions)

    append = add_partitions

    def remove_partitions(self, delete_scope=None, predicates=None) -> "TableWriter":

        if delete_scope:
            # raise a warning in the future and migrate to a better interface
            pass
        elif predicates:
            # It should be possible to pass the delete scope also via the predicates if one
            # allows only partition cols
            raise NotImplementedError()

        return self.copy(delete_scope=delete_scope)


class BaseTableWriter(TableWriter):
    def commit(self, compute=True):
        if compute:
            from kartothek.io.eager import update_dataset_from_dataframes

            func = update_dataset_from_dataframes
        else:
            from kartothek.io.iter import update_dataset_from_dataframes__iter

            func = update_dataset_from_dataframes__iter
        return func(
            df_list=[{self.table.name: part} for part in self.append_partitions],
            factory=self.table._factory,
            delete_scope=self.delete_scope,
            metadata=self.metadata,
            df_serializer=self.df_serializer,
            default_metadata_version=self.metadata_version,
            sort_partitions_by=self.sort_partitions_by,
            secondary_indices=self.secondary_indices,
        )


@attr.s(auto_attribs=True)
class DaskDataFrameTableWriter(TableWriter):
    pass


@attr.s(auto_attribs=True)
class DaskBagTableWriter(TableWriter):
    pass
