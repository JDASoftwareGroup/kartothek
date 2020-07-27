import copy
import logging
import re
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

import pandas as pd
import pyarrow as pa
import simplejson
from simplekv import KeyValueStore

import kartothek.core._time
import kartothek.core._zmsgpack as msgpack
from kartothek.core import naming
from kartothek.core._compat import load_json
from kartothek.core._mixins import CopyMixin
from kartothek.core.common_metadata import SchemaWrapper, read_schema_metadata
from kartothek.core.docs import default_docs
from kartothek.core.index import (
    ExplicitSecondaryIndex,
    IndexBase,
    PartitionIndex,
    filter_indices,
)
from kartothek.core.naming import EXTERNAL_INDEX_SUFFIX, PARQUET_FILE_SUFFIX
from kartothek.core.partition import Partition
from kartothek.core.urlencode import decode_key, quote_indices
from kartothek.core.utils import verify_metadata_version
from kartothek.serialization import PredicatesType, columns_in_predicates

_logger = logging.getLogger(__name__)

TableMetaType = Dict[str, SchemaWrapper]


def _validate_uuid(uuid: str) -> bool:
    return re.match(r"[a-zA-Z0-9+\-_]+$", uuid) is not None


def to_ordinary_dict(dct: Dict) -> Dict:
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            new_dct[key] = to_ordinary_dict(value)
        else:
            new_dct[key] = value
    return new_dct


T = TypeVar("T", bound="DatasetMetadataBase")


class DatasetMetadataBase(CopyMixin):
    def __init__(
        self,
        uuid: str,
        partitions: Optional[Dict[str, Partition]] = None,
        metadata: Optional[Dict] = None,
        indices: Optional[Dict[str, IndexBase]] = None,
        metadata_version: int = naming.DEFAULT_METADATA_VERSION,
        explicit_partitions: bool = True,
        partition_keys: Optional[List[str]] = None,
        table_meta: Optional[Dict[str, SchemaWrapper]] = None,
    ):
        if not _validate_uuid(uuid):
            raise ValueError("UUID contains illegal character")
        self.metadata_version = metadata_version
        self.uuid = uuid
        self.partitions = partitions if partitions else {}
        self.metadata = metadata if metadata else {}
        self.indices = indices if indices else {}
        # explicit partitions means that the partitions are defined in the
        # metadata.json file (in contrast to implicit partitions that are
        # derived from the partition key names)
        self.explicit_partitions = explicit_partitions

        self.partition_keys = partition_keys or []
        self.table_meta = table_meta if table_meta else {}

        _add_creation_time(self)
        super(DatasetMetadataBase, self).__init__()

    def __eq__(self, other: Any) -> bool:
        # Enforce dict comparison at the places where we only
        # care about content, not order.
        if self.uuid != other.uuid:
            return False
        if to_ordinary_dict(self.partitions) != to_ordinary_dict(other.partitions):
            return False
        if to_ordinary_dict(self.metadata) != to_ordinary_dict(other.metadata):
            return False
        if self.indices != other.indices:
            return False
        if self.explicit_partitions != other.explicit_partitions:
            return False
        if self.partition_keys != other.partition_keys:
            return False
        if self.table_meta != other.table_meta:
            return False
        return True

    @property
    def primary_indices_loaded(self) -> bool:
        if not self.partition_keys:
            return False
        for pkey in self.partition_keys:
            if pkey not in self.indices:
                return False
        return True

    @property
    def tables(self) -> List[str]:
        if self.table_meta:
            return list(self.table_meta.keys())
        elif self.partitions:
            return [tab for tab in list(self.partitions.values())[0].files]
        else:
            return []

    @property
    def index_columns(self) -> Set[str]:
        return set(self.indices.keys()).union(self.partition_keys)

    @property
    def secondary_indices(self) -> Dict[str, ExplicitSecondaryIndex]:
        return {
            col: ind
            for col, ind in self.indices.items()
            if isinstance(ind, ExplicitSecondaryIndex)
        }

    @staticmethod
    def exists(uuid: str, store: KeyValueStore) -> bool:
        """
        Check if  a dataset exists in a storage

        Parameters
        ----------
        uuid: str or unicode
            UUID of the dataset.
        store: Object
            Object that implements the .get method for file/object loading.

        Returns
        -------
        exists: bool
            Whether a metadata file could be found.
        """
        key = naming.metadata_key_from_uuid(uuid)

        if key in store:
            return True

        key = naming.metadata_key_from_uuid(uuid, format="msgpack")
        return key in store

    @staticmethod
    def storage_keys(uuid: str, store: KeyValueStore) -> List[str]:
        """
        Retrieve all keys that belong to the given dataset.

        Parameters
        ----------
        uuid:
            UUID of the dataset.
        store:
            Object that implements the .iter_keys method for key retrieval loading.

        Returns
        -------
        keys:
            Sorted list of storage keys.
        """
        start_markers = ["{}.".format(uuid), "{}/".format(uuid)]
        return list(
            sorted(
                k
                for k in store.iter_keys(uuid)
                if any(k.startswith(marker) for marker in start_markers)
            )
        )

    def to_dict(self) -> Dict:
        dct = OrderedDict(
            [
                (naming.METADATA_VERSION_KEY, self.metadata_version),
                (naming.UUID_KEY, self.uuid),
            ]
        )
        if self.indices:
            dct["indices"] = {
                k: v.to_dict()
                if v.loaded
                else v.index_storage_key
                if isinstance(v, ExplicitSecondaryIndex)
                else {}
                for k, v in self.indices.items()
            }
        if self.metadata:
            dct["metadata"] = self.metadata
        if self.partitions or self.explicit_partitions:
            dct["partitions"] = {
                label: partition.to_dict()
                for label, partition in self.partitions.items()
            }

        if self.partition_keys is not None:
            dct["partition_keys"] = self.partition_keys
        # don't preserve table_meta, since there is no JSON-compatible way (yet)

        return dct

    def to_json(self) -> bytes:
        return simplejson.dumps(self.to_dict()).encode("utf-8")

    def to_msgpack(self) -> bytes:
        return msgpack.packb(self.to_dict())

    def load_index(self: T, column: str, store: KeyValueStore) -> T:
        """
        Load an index into memory.

        Note: External indices need to be preloaded before they can be queried.

        Parameters
        ----------
        column: str
            Name of the column for which the index should be loaded.
        store: Object
            Object that implements the .get method for file/object loading.

        Returns
        -------
        dataset_metadata: :class:`~kartothek.core.dataset.DatasetMetadata`
            Mutated metadata object with the loaded index.
        """
        if self.partition_keys and column in self.partition_keys:
            return self.load_partition_indices()

        if column not in self.indices:
            raise KeyError("No index specified for column '{}'".format(column))

        index = self.indices[column]
        if index.loaded or not isinstance(index, ExplicitSecondaryIndex):
            return self

        loaded_index = index.load(store=store)
        if not self.explicit_partitions:
            col_loaded_index = filter_indices(
                {column: loaded_index}, self.partitions.keys()
            )
        else:
            col_loaded_index = {column: loaded_index}
        indices = dict(self.indices, **col_loaded_index)
        return self.copy(indices=indices)

    def load_all_indices(
        self: T, store: KeyValueStore, load_partition_indices: bool = True
    ) -> T:
        """
        Load all registered indices into memory.

        Note: External indices need to be preloaded before they can be queried.

        Parameters
        ----------
        store: Object
            Object that implements the .get method for file/object loading.
        load_partition_indices: bool
            Flag if filename indices should be loaded. Default is True.

        Returns
        -------
        dataset_metadata: :class:`~kartothek.core.dataset.DatasetMetadata`
            Mutated metadata object with the loaded indices.
        """
        indices = {
            column: index.load(store)
            if isinstance(index, ExplicitSecondaryIndex)
            else index
            for column, index in self.indices.items()
        }
        ds = self.copy(indices=indices)

        if load_partition_indices:
            ds = ds.load_partition_indices()
        return ds

    def query(self, indices: List[IndexBase] = None, **kwargs) -> List[str]:
        """
        Query the dataset for partitions that contain specific values. Lookup is performed
        using the embedded and loaded external indices. Additional indices need to operate
        on the same partitions that the dataset contains, otherwise an empty list will be
        returned (the query method only restricts the set of partition keys using the indices).

        Parameters
        ----------
        indices:
            List of optional additional indices.
        **kwargs:
            Map of columns and values.

        Returns
        -------
            List of keys of partitions that contain the queries values in the respective columns.
        """
        candidate_set = set(self.partitions.keys())

        additional_indices = indices if indices else {}
        combined_indices = dict(
            self.indices, **{index.column: index for index in additional_indices}
        )

        for column, value in kwargs.items():
            if column in combined_indices:
                candidate_set &= set(combined_indices[column].query(value))

        return list(candidate_set)

    def load_partition_indices(self: T) -> T:
        """
        Load all filename encoded indices into RAM. File encoded indices can be extracted from datasets with partitions
        stored in a format like

        .. code::

            `dataset_uuid/table/IndexCol=IndexValue/SecondIndexCol=Value/partition_label.parquet`

        Which results in an in-memory index holding the information

        .. code::

            {
                "IndexCol": {
                    IndexValue: ["partition_label"]
                },
                "SecondIndexCol": {
                    Value: ["partition_label"]
                }
            }

        """
        if self.primary_indices_loaded:
            return self

        indices = _construct_dynamic_index_from_partitions(
            partitions=self.partitions,
            table_meta=self.table_meta,
            default_dtype=pa.string() if self.metadata_version == 3 else None,
            partition_keys=self.partition_keys,
        )
        combined_indices = self.indices.copy()
        combined_indices.update(indices)
        return self.copy(indices=combined_indices)

    @default_docs
    def get_indices_as_dataframe(
        self,
        columns: Optional[List[str]] = None,
        date_as_object: bool = True,
        predicates: PredicatesType = None,
    ):
        """
        Converts the dataset indices to a pandas dataframe and filter relevant indices by `predicates`.

        For a dataset with indices on columns `column_a` and `column_b` and three partitions,
        the dataset output may look like

        .. code::

                    column_a column_b
            part_1         1        A
            part_2         2        B
            part_3         3     None

        Parameters
        ----------
        """
        if columns is None:
            columns = sorted(self.indices.keys())
        elif columns == []:
            return pd.DataFrame(index=self.partitions)

        columns_to_scan = columns[:]
        if predicates:
            predicate_columns = columns_in_predicates(predicates)
            columns_to_scan = sorted(
                (predicate_columns & self.indices.keys()) | set(columns)
            )

            dfs = (
                self._evaluate_conjunction(
                    columns=columns_to_scan,
                    predicates=[conjunction],
                    date_as_object=date_as_object,
                )
                for conjunction in predicates
            )

            df = pd.concat(dfs)
            index_name = df.index.name
            df = (
                df.loc[:, columns].reset_index().drop_duplicates().set_index(index_name)
            )
        else:
            df = self._evaluate_conjunction(
                columns=columns_to_scan, predicates=None, date_as_object=date_as_object,
            )
        return df

    def _evaluate_conjunction(self, columns, predicates, date_as_object):
        """
        Evaluate all predicates related to `columns` to "AND".

        Parameters
        ----------
        columns:
            A list of all columns, including query and index columns.
        predicates: list of list of tuple[str, str, Any]
            Optional list of predicates, like [[('x', '>', 0), ...], that are used
            to filter the resulting DataFrame, possibly using predicate pushdown,
            if supported by the file format.
            This parameter is not compatible with filter_query.

            Predicates are expressed in disjunctive normal form (DNF). This means
            that the innermost tuple describes a single column predicate. These
            inner predicates are all combined with a conjunction (AND) into a
            larger predicate. The most outer list then combines all predicates
            with a disjunction (OR). By this, we should be able to express all
            kinds of predicates that are possible using boolean logic.

            Available operators are: `==`, `!=`, `<=`, `>=`, `<`, `>` and `in`.
        dates_as_object: bool
            Load pyarrow.date{32,64} columns as ``object`` columns in Pandas
            instead of using ``np.datetime64`` to preserve their type. While
            this improves type-safety, this comes at a performance cost.

        Returns
        -------
        df_result:
            A DataFrame containing all indices for which `predicates` holds true.
        """
        non_index_columns = set(columns) - self.indices.keys()
        if non_index_columns:
            if non_index_columns & set(self.partition_keys):
                raise RuntimeError(
                    "Partition indices not loaded. Please call `DatasetMetadata.load_partition_indices` first."
                )
            raise ValueError(
                "Unknown index columns: {}".format(", ".join(sorted(non_index_columns)))
            )
        dfs = []
        for col in columns:
            df = pd.DataFrame(
                self.indices[col].as_flat_series(
                    partitions_as_index=True,
                    date_as_object=date_as_object,
                    predicates=predicates,
                )
            )
            dfs.append(df)
        # dfs contains one df per index column. Each df stores indices filtered by `predicates` for each column.
        # Performing an inner join on these dfs yields the resulting "AND" evaluation for all of these predicates.
        # We start joining with the smallest dataframe, therefore the sorting.
        dfs_sorted = sorted(dfs, key=len)
        df_result = dfs_sorted.pop(0)
        for df in dfs_sorted:
            df_result = df_result.merge(
                df, left_index=True, right_index=True, copy=False
            )
        return df_result


class DatasetMetadata(DatasetMetadataBase):
    """
    Containing holding all metadata of the dataset.
    """

    def __repr__(self):
        return (
            "DatasetMetadata(uuid={uuid}, "
            "tables={tables}, "
            "partition_keys={partition_keys}, "
            "metadata_version={metadata_version}, "
            "indices={indices}, "
            "explicit_partitions={explicit_partitions})"
        ).format(
            uuid=self.uuid,
            tables=self.tables,
            partition_keys=self.partition_keys,
            metadata_version=self.metadata_version,
            indices=list(self.indices.keys()),
            explicit_partitions=self.explicit_partitions,
        )

    @staticmethod
    def load_from_buffer(
        buf, store: KeyValueStore, format: str = "json"
    ) -> "DatasetMetadata":
        """
        Load a dataset from a (string) buffer.

        Parameters
        ----------
        buf:
            Input to be parsed.
        store:
            Object that implements the .get method for file/object loading.

        Returns
        -------
        dataset_metadata:
            Parsed metadata.
        """
        if format == "json":
            metadata = load_json(buf)
        elif format == "msgpack":
            metadata = msgpack.unpackb(buf)
        return DatasetMetadata.load_from_dict(metadata, store)

    @staticmethod
    def load_from_store(
        uuid: str,
        store: KeyValueStore,
        load_schema: bool = True,
        load_all_indices: bool = False,
    ) -> "DatasetMetadata":
        """
        Load a dataset from a storage

        Parameters
        ----------
        uuid: str or unicode
            UUID of the dataset.
        store: Object
            Object that implements the .get method for file/object loading.
        load_schema: bool
            Load table schema
        load_all_indices: bool
            Load all registered indices into memory.

        Returns
        -------
        dataset_metadata: :class:`~kartothek.core.dataset.DatasetMetadata`
            Parsed metadata.
        """
        key1 = naming.metadata_key_from_uuid(uuid)
        try:
            value = store.get(key1)
            metadata = load_json(value)
        except KeyError:
            key2 = naming.metadata_key_from_uuid(uuid, format="msgpack")
            try:
                value = store.get(key2)
                metadata = msgpack.unpackb(value)
            except KeyError:
                raise KeyError(
                    "Dataset does not exist. Tried {} and {}".format(key1, key2)
                )

        ds = DatasetMetadata.load_from_dict(metadata, store, load_schema=load_schema)
        if load_all_indices:
            ds = ds.load_all_indices(store)
        return ds

    @staticmethod
    def load_from_dict(dct: Dict, store: KeyValueStore, load_schema: bool = True):
        """
        Load dataset metadata from a dictionary and resolve any external includes.

        Parameters
        ----------
        dct: dict
        store: Object
            Object that implements the .get method for file/object loading.
        load_schema: bool
            Load table schema

        Returns
        -------
        dataset_metadata: :class:`~kartothek.core.dataset.DatasetMetadata`
            Parsed metadata.
        """
        # Use copy here to get an OrderedDict
        metadata = copy.copy(dct)

        if "metadata" not in metadata:
            metadata["metadata"] = OrderedDict()

        metadata_version = dct[naming.METADATA_VERSION_KEY]
        dataset_uuid = dct[naming.UUID_KEY]
        explicit_partitions = "partitions" in metadata
        storage_keys = None
        if not explicit_partitions:
            storage_keys = DatasetMetadata.storage_keys(dataset_uuid, store)
            partitions = _load_partitions_from_filenames(
                store=store,
                storage_keys=storage_keys,
                metadata_version=metadata_version,
            )
            metadata["partitions"] = partitions

        if metadata["partitions"]:
            tables = [tab for tab in list(metadata["partitions"].values())[0]["files"]]
        else:
            table_set = set()
            if storage_keys is None:
                storage_keys = DatasetMetadata.storage_keys(dataset_uuid, store)
            for key in storage_keys:
                if key.endswith(naming.TABLE_METADATA_FILE):
                    table_set.add(key.split("/")[1])
            tables = list(table_set)

        table_meta = {}
        if load_schema:
            for table in tables:
                table_meta[table] = read_schema_metadata(
                    dataset_uuid=dataset_uuid, store=store, table=table
                )

        metadata["table_meta"] = table_meta

        if "partition_keys" not in metadata:
            metadata["partition_keys"] = _get_partition_keys_from_partitions(
                metadata["partitions"]
            )

        return DatasetMetadata.from_dict(
            metadata, explicit_partitions=explicit_partitions
        )

    @staticmethod
    def from_buffer(buf: str, format: str = "json", explicit_partitions: bool = True):
        if format == "json":
            metadata = load_json(buf)
        else:
            metadata = msgpack.unpackb(buf)
        return DatasetMetadata.from_dict(
            metadata, explicit_partitions=explicit_partitions
        )

    @staticmethod
    def from_dict(dct: Dict, explicit_partitions: bool = True):
        """
        Load dataset metadata from a dictionary.

        This must have no external references. Otherwise use ``load_from_dict``
        to have them resolved automatically.
        """

        # Use the builder class for reconstruction to have a single point for metadata version changes
        builder = DatasetMetadataBuilder(
            uuid=dct[naming.UUID_KEY],
            metadata_version=dct[naming.METADATA_VERSION_KEY],
            explicit_partitions=explicit_partitions,
            partition_keys=dct.get("partition_keys", None),
            table_meta=dct.get("table_meta", None),
        )

        for key, value in dct.get("metadata", {}).items():
            builder.add_metadata(key, value)
        for partition_label, part_dct in dct.get("partitions", {}).items():
            builder.add_partition(
                partition_label, Partition.from_dict(partition_label, part_dct)
            )
        for column, index_dct in dct.get("indices", {}).items():
            if isinstance(index_dct, IndexBase):
                builder.add_embedded_index(column, index_dct)
            else:
                builder.add_embedded_index(
                    column, ExplicitSecondaryIndex.from_v2(column, index_dct)
                )
        return builder.to_dataset()


def _get_type_from_meta(
    table_meta: Optional[Dict[str, SchemaWrapper]],
    column: str,
    default: Optional[pa.DataType],
) -> pa.DataType:
    # use first schema that provides type information, since write path should ensure that types are normalized and
    # equal
    if table_meta is not None:
        for schema in table_meta.values():
            if column not in schema.names:
                continue
            idx = schema.get_field_index(column)
            return schema[idx].type

    if default is not None:
        return default

    raise ValueError(
        'Cannot find type information for partition column "{}"'.format(column)
    )


def _empty_partition_indices(
    partition_keys: List[str], table_meta: TableMetaType, default_dtype: pa.DataType
):
    indices = {}
    for col in partition_keys:
        arrow_type = _get_type_from_meta(table_meta, col, default_dtype)
        indices[col] = PartitionIndex(column=col, index_dct={}, dtype=arrow_type)
    return indices


def _construct_dynamic_index_from_partitions(
    partitions: Dict[str, Partition],
    table_meta: TableMetaType,
    default_dtype: pa.DataType,
    partition_keys: List[str],
) -> Dict[str, PartitionIndex]:
    if len(partitions) == 0:
        return _empty_partition_indices(partition_keys, table_meta, default_dtype)

    def _get_files(part):
        if isinstance(part, dict):
            return part["files"]
        else:
            return part.files

    # We exploit the fact that all tables are partitioned equally.
    first_partition = next(
        iter(partitions.values())
    )  # partitions is NOT empty here, see check above
    first_partition_files = _get_files(first_partition)
    if not first_partition_files:
        return _empty_partition_indices(partition_keys, table_meta, default_dtype)
    key_table = next(iter(first_partition_files.keys()))
    storage_keys = (
        (key, _get_files(part)[key_table]) for key, part in partitions.items()
    )

    _key_indices: Dict[str, Dict[str, Set[str]]] = defaultdict(_get_empty_index)
    depth_indices = None
    for partition_label, key in storage_keys:
        _, _, indices, file_ = decode_key(key)
        if (
            file_ is not None
            and key.endswith(PARQUET_FILE_SUFFIX)
            and not key.endswith(EXTERNAL_INDEX_SUFFIX)
        ):
            depth_indices = _check_index_depth(indices, depth_indices)
            for column, value in indices:
                _key_indices[column][value].add(partition_label)
    new_indices = {}
    for col, index_dct in _key_indices.items():
        arrow_type = _get_type_from_meta(table_meta, col, default_dtype)

        # convert defaultdicts into dicts
        new_indices[col] = PartitionIndex(
            column=col,
            index_dct={k1: list(v1) for k1, v1 in index_dct.items()},
            dtype=arrow_type,
        )
    return new_indices


def _get_partition_label(indices, filename, metadata_version):
    return "/".join(
        quote_indices(indices) + [filename.replace(PARQUET_FILE_SUFFIX, "")]
    )


def _check_index_depth(indices, depth_indices):
    if depth_indices is not None and len(indices) != depth_indices:
        raise RuntimeError(
            "Unknown file structure encountered. "
            "Depth of filename indices is not equal for all partitions."
        )
    return len(indices)


def _get_partition_keys_from_partitions(partitions):
    if len(partitions):
        part = next(iter(partitions.values()))
        files_dct = part["files"]
        if files_dct:
            key = next(iter(files_dct.values()))
            _, _, indices, _ = decode_key(key)
            if indices:
                return [tup[0] for tup in indices]
    return None


def _load_partitions_from_filenames(store, storage_keys, metadata_version):
    partitions = defaultdict(_get_empty_partition)
    depth_indices = None
    for key in storage_keys:
        dataset_uuid, table, indices, file_ = decode_key(key)
        if file_ is not None and file_.endswith(PARQUET_FILE_SUFFIX):
            # valid key example:
            # <uuid>/<table>/<column_0>=<value_0>/.../<column_n>=<value_n>/part_label.parquet
            depth_indices = _check_index_depth(indices, depth_indices)
            partition_label = _get_partition_label(indices, file_, metadata_version)
            partitions[partition_label]["files"][table] = key
    return partitions


def _get_empty_partition():
    return {"files": {}, "metadata": {}}


def _get_empty_index():
    return defaultdict(set)


def create_partition_key(dataset_uuid, table, index_values, filename="data"):
    """
    Create partition key for a kartothek partition

    Parameters
    ----------
    dataset_uuid: str
    table: str
    index_values: list of tuples str:str
    filename: str

    Example:
        create_partition_key('my-uuid', 'testtable',
            [('index1', 'value1'), ('index2', 'value2')])

        returns 'my-uuid/testtable/index1=value1/index2=value2/data'
    """
    key_components = [dataset_uuid, table]
    index_path = quote_indices(index_values)
    key_components.extend(index_path)
    key_components.append(filename)
    key = "/".join(key_components)
    return key


class DatasetMetadataBuilder(CopyMixin):
    """
    Incrementally build up a dataset.

    In constrast to a :class:`kartothek.core.dataset.DatasetMetadata` instance,
    this object is mutable and may not be a full dataset (e.g. partitions don't
    need to be fully materialised).
    """

    def __init__(
        self,
        uuid: str,
        metadata_version=naming.DEFAULT_METADATA_VERSION,
        explicit_partitions=True,
        partition_keys=None,
        table_meta=None,
    ):
        verify_metadata_version(metadata_version)

        self.uuid = uuid
        self.metadata: Dict = OrderedDict()
        self.indices: Dict[str, IndexBase] = OrderedDict()
        self.metadata_version = metadata_version
        self.partitions: Dict[str, Partition] = OrderedDict()
        self.partition_keys = partition_keys
        self.table_meta = table_meta
        self.explicit_partitions = explicit_partitions

        _add_creation_time(self)
        super(DatasetMetadataBuilder, self).__init__()

    @staticmethod
    def from_dataset(dataset):
        dataset = copy.deepcopy(dataset)

        ds_builder = DatasetMetadataBuilder(
            uuid=dataset.uuid,
            metadata_version=dataset.metadata_version,
            explicit_partitions=dataset.explicit_partitions,
            partition_keys=dataset.partition_keys,
            table_meta=dataset.table_meta,
        )

        ds_builder.metadata = dataset.metadata
        ds_builder.indices = dataset.indices
        ds_builder.partitions = dataset.partitions
        ds_builder.tables = dataset.tables
        return ds_builder

    def add_partition(self, name, partition):
        """
        Add an (embedded) Partition.

        Parameters
        ----------
        name: str
            Identifier of the partition.
        partition: :class:`kartothek.core.partition.Partition`
            The partition to add.
        """
        self.partitions[name] = partition
        return self

    # TODO: maybe remove
    def add_embedded_index(self, column, index):
        """
        Embed an index into the metadata.

        Parameters
        ----------
        column: str
            Name of the indexed column
        index: kartothek.core.index.IndexBase
            The actual index object
        """

        if column != index.column:
            # TODO Deprecate the column argument and take the column name directly from the index.
            raise RuntimeError(
                "The supplied index is not compatible with the supplied index."
            )

        self.indices[column] = index

    def add_external_index(self, column, filename=None):
        """
        Add a reference to an external index.

        Parameters
        ----------
        column: str
            Name of the indexed column

        Returns
        -------
        storage_key: str
            The location where the external index should be stored.
        """
        if filename is None:
            filename = "{uuid}.{column_name}".format(uuid=self.uuid, column_name=column)
            filename += naming.EXTERNAL_INDEX_SUFFIX
        self.indices[column] = ExplicitSecondaryIndex(
            column, index_storage_key=filename
        )
        return filename

    def add_metadata(self, key, value):
        """
        Add arbitrary key->value metadata.

        Parameters
        ----------
        key: str
        value: str
        """
        self.metadata[key] = value

    def to_dict(self):
        """
        Render the dataset to a dict.

        Returns
        -------

        """
        factory = type(self.metadata)
        dct = factory(
            [
                (naming.METADATA_VERSION_KEY, self.metadata_version),
                (naming.UUID_KEY, self.uuid),
            ]
        )
        if self.indices:
            dct["indices"] = {}
            for column, index in self.indices.items():
                if isinstance(index, str):
                    dct["indices"][column] = index
                elif index.loaded:
                    dct["indices"][column] = index.to_dict()
                else:
                    dct["indices"][column] = index.index_storage_key
        if self.metadata:
            dct["metadata"] = self.metadata

        if self.explicit_partitions:
            dct["partitions"] = factory()
            for label, partition in self.partitions.items():
                part_dict = partition.to_dict()
                dct["partitions"][label] = part_dict

        if self.partition_keys is not None:
            dct["partition_keys"] = self.partition_keys
        # don't preserve table_meta, since there is no JSON-compatible way (yet)
        return dct

    def to_json(self):
        """
        Render the dataset to JSON.

        Returns
        -------
        storage_key: str
            The path where this metadata should be placed in the storage.
        dataset_json: str
            The rendered JSON for this dataset.
        """
        return (
            naming.metadata_key_from_uuid(self.uuid),
            simplejson.dumps(self.to_dict()).encode("utf-8"),
        )

    def to_msgpack(self) -> Tuple[str, bytes]:
        """
        Render the dataset to msgpack.

        Returns
        -------
        storage_key: str
            The path where this metadata should be placed in the storage.
        dataset_json: str
            The rendered JSON for this dataset.
        """
        return (
            naming.metadata_key_from_uuid(self.uuid, format="msgpack"),
            msgpack.packb(self.to_dict()),
        )

    def to_dataset(self) -> DatasetMetadata:
        return DatasetMetadata(
            uuid=self.uuid,
            partitions=self.partitions,
            metadata=self.metadata,
            indices=self.indices,
            metadata_version=self.metadata_version,
            explicit_partitions=self.explicit_partitions,
            partition_keys=self.partition_keys,
            table_meta=self.table_meta,
        )


def _add_creation_time(
    dataset_object: Union[DatasetMetadataBase, DatasetMetadataBuilder]
):
    if "creation_time" not in dataset_object.metadata:
        creation_time = kartothek.core._time.datetime_utcnow().isoformat()
        dataset_object.metadata["creation_time"] = creation_time
