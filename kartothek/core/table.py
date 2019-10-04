from typing import Any, Callable, Dict, List, Optional

import attr
import dask
import dask.bag as db
import dask.dataframe as dd
import pandas as pd

from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.serialization import ParquetSerializer

from .common_metadata import SchemaWrapper
from .factory import DatasetFactory


class Table:
    def __init__(
        self, dataset_uuid: str, store_factory: Callable, name: str = SINGLE_TABLE
    ):
        self.name = name
        self.dataset_uuid = dataset_uuid
        self.store_factory = store_factory
        self._factory = DatasetFactory(dataset_uuid, store_factory)

    @staticmethod
    def from_factory(self, factory: DatasetFactory, name=SINGLE_TABLE) -> "Table":
        inst = Table("__dummy__", lambda x: None, name=name)
        inst._factory = factory
        return inst

    @property
    def schema(self) -> SchemaWrapper:
        return self._factory.table_meta[self.name]

    def read(self) -> "TableReader":
        return TableReader(table=self)

    def write(self) -> "TableWriter":
        return TableWriter(table=self)

    def __getattr__(self, name: str):
        # __getattr__ should only be called if the attribute cannot be found. if the
        # attribute is None, it still falls back to this call
        factory = getattr(self, "_factory")
        return getattr(factory, name)


class TableOperation:
    def to_dict(self):
        options = self.options()
        options.update({"table": self.table})
        return options

    def reset(self):
        return type(self)(self.table)

    def bind(self, **kwargs):
        dct = self.to_dict()
        dct.update(kwargs)
        return type(self)(**dct)


class TableReader(TableOperation):
    def __init__(
        self,
        table: Table,
        *,
        predicates: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        dispatch_by: Optional[List[str]] = None,
        predicate_pushdown_to_io: Optional[bool] = None,
        categories: Optional[List[str]] = None,
        dates_as_object: Optional[bool] = False,
    ):
        self.table = table
        self._predicates = predicates
        self._columns = columns
        self._dispatch_by = dispatch_by
        self._predicate_pushdown_to_io = predicate_pushdown_to_io
        self._categories = categories
        self._dates_as_object = dates_as_object

    def options(self):
        return {
            "predicates": self._predicates,
            "columns": self._columns,
            "dispatch_by": self._dispatch_by,
            "predicate_pushdown_to_io": self._predicate_pushdown_to_io,
            "categories": self._categories,
            "dates_as_object": self._dates_as_object,
        }

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
            predicates=self._predicates,
            columns=self._columns,
            table=self.table.name,
            categoricals=self._categories,
            dates_as_object=self._dates_as_object,
        )

    def categories(self, categories):
        if isinstance(categories, dict):
            categories = categories[self.table.name]
        return self.bind(categories=categories)

    def dates_as_object(self, dates_as_object):
        return self.bind(dates_as_object=dates_as_object)

    def predicate_pushdown_to_io(self, value):
        return self.bind(predicate_pushdown_to_io=value)

    def filter(self, predicates) -> "TableReader":
        return self.bind(predicates=predicates)

    def columns(self, columns):
        return self.bind(columns=columns)

    def dispatch_by(self, columns) -> "TableReader":
        return self.bind(dispatch_by=columns)


class TableWriter(TableOperation):
    def __init__(
        self,
        table: Table,
        *,
        secondary_indices: Optional[List[str]] = None,
        partition_on: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        metadata_version: Optional[str] = None,
        sort_by: Optional[str] = None,
        append_partitions: Optional[List] = None,
        delete_scope: Optional[List[Dict]] = None,
        allow_override: Optional[bool] = False,
        compression: Optional[str] = "zstd",
        parquet_chunksize: Optional[int] = None,
    ):
        self.table = table
        self._secondary_indices = secondary_indices
        self._partition_on = partition_on
        self._metadata = metadata
        self._metadata_version = metadata_version
        self._sort_by = sort_by
        self._append_partitions = append_partitions
        self._delete_scope = delete_scope
        self._allow_override = allow_override
        self._compression = compression
        self._parquet_chunksize = parquet_chunksize

    def options(self):
        return {
            "secondary_indices": self._secondary_indices,
            "partition_on": self._partition_on,
            "metadata": self._metadata,
            "metadata_version": self._metadata_version,
            "sort_by": self._sort_by,
            "append_partitions": self._append_partitions,
            "delete_scope": self._delete_scope,
            "allow_override": self._allow_override,
            "compression": self._compression,
            "parquet_chunksize": self._parquet_chunksize,
        }

    @property
    def df_serializer(self):
        return ParquetSerializer(
            chunk_size=self._parquet_chunksize, compression=self._compression
        )

    @property
    def TYPED_TABLEWRITER(self):
        return {
            dd.DataFrame: DaskDataFrameTableWriter,
            db.Bag: DaskDataFrameTableWriter,
        }

    def as_parquet(self):
        return

    def commit(self, compute=True):
        raise NotImplementedError

    def index_on(self, columns):
        return self.bind(secondary_indices=columns)

    def partition_on(self, columns):
        return self.bind(partition_on=columns)

    def sort_by(self, columns):
        return self.bind(sort_by=columns)

    def add_metadata(self, metadata):
        return self.bind(metadata=metadata)

    def _get_contructor_from_partitions(self, obj):
        if isinstance(obj, list) and obj:
            return self._get_contructor_from_partitions(obj[0])
        else:
            return self.TYPED_TABLEWRITER.get(type(obj), BaseTableWriter)

    def add_partitions(self, partitions=None, ddf=None) -> "TableWriter":
        writer_class = self._get_contructor_from_partitions(partitions)

        instance = writer_class(table=self.table)
        return instance.bind(append_partitions=partitions)

    append = add_partitions

    def remove_partitions(self, delete_scope=None, predicates=None) -> "TableWriter":

        if delete_scope:
            # raise a warning in the future and migrate to a better interface
            pass
        elif predicates:
            # It should be possible to pass the delete scope also via the predicates if one
            # allows only partition cols
            raise NotImplementedError()

        return self.bind(delete_scope=delete_scope)


class BaseTableWriter(TableWriter):
    def commit(self, compute=True):
        if compute:
            from kartothek.io.eager import update_dataset_from_dataframes

            func = update_dataset_from_dataframes
        else:
            from kartothek.io.iter import update_dataset_from_dataframes__iter

            func = update_dataset_from_dataframes__iter

        return func(
            df_list=[{self.table.name: part} for part in self._append_partitions],
            factory=self.table._factory,
            delete_scope=self._delete_scope,
            metadata=self._metadata,
            df_serializer=self.df_serializer,
            default_metadata_version=self._metadata_version,
            sort_partitions_by=self._sort_by,
            secondary_indices=self._secondary_indices,
        )


@attr.s(auto_attribs=True)
class DaskDataFrameTableWriter(TableWriter):
    pass


@attr.s(auto_attribs=True)
class DaskBagTableWriter(TableWriter):
    pass
