# -*- coding: utf-8 -*-

import logging
from copy import copy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import kartothek.core._time
from kartothek.core import naming
from kartothek.core._mixins import CopyMixin
from kartothek.core.common_metadata import normalize_type
from kartothek.core.urlencode import quote
from kartothek.serialization import filter_array_like
from kartothek.serialization._parquet import _fix_pyarrow_07992_table

_logger = logging.getLogger(__name__)

_PARTITION_COLUMN_NAME = "partition"


class IndexBase(CopyMixin):
    """
    Base class for all index types.


    Parameters
    ----------
    column: string
        Name of the column this index is for.
    index_dct: Union[None, Dict[String, List[String]]]
        Mapping from index values to partition labels
    dtype: Union[None, pyarrow.Type]
        Type of index. If left out and ``index_dct`` is present, this will be inferred.
    normalize_dtype: bool
        Normalize type information and values within ``index_dct``. The user may disable this when it the index was
        already normalized, e.g. when the index python objects gets copied, or when the index data is restored from a
        parquet file that was written by a trusted write path.
    """

    def __init__(self, column, index_dct=None, dtype=None, normalize_dtype=True):
        if column == _PARTITION_COLUMN_NAME:
            raise ValueError(
                "Cannot create an index for column {} due to an internal implementation conflict. "
                "Please contact a kartothek maintainer if you receive this error message".format(
                    column
                )
            )
        if (dtype is None) and index_dct:
            # do dtype given but index_dct is present => auto-derive dtype
            table = _index_dct_to_table(index_dct, column)
            schema = table.schema
            dtype = schema[0].type

        if dtype is not None:
            if pa.types.is_nested(dtype):
                raise NotImplementedError("Indices w/ nested types are not supported")
            if pa.types.is_null(dtype):
                raise NotImplementedError("Indices w/ null/NA type are not supported")
            if normalize_dtype:
                dtype, _t_pd, _t_np, _metadata = normalize_type(dtype, None, None, None)

        self.column = column
        self.dtype = dtype
        self.creation_time = kartothek.core._time.datetime_utcnow()

        if index_dct is None:
            self.index_dct = None
        else:
            if normalize_dtype:
                # index_dct may be from an untrusted or weakly typed source, so we need to normalize and fuse values
                self.index_dct = {}
                n_collisions = 0
                for value, partitions in index_dct.items():
                    value = IndexBase.normalize_value(self.dtype, value)
                    if value not in self.index_dct:
                        self.index_dct[value] = copy(partitions)
                    else:
                        existing = self.index_dct[value]
                        self.index_dct[value] += [
                            part for part in partitions if part not in existing
                        ]
                        n_collisions += 1

                if n_collisions:
                    _logger.warning(
                        (
                            "Value normalization for index column {} resulted in {} collision(s). Kartothek merged "
                            "the affected partition lists, but you may want to check if this was desired."
                        ).format(column, n_collisions)
                    )
            else:
                # data comes from a trusted source (e.g. an index that we've already preserved and are now reading), so
                # use the fast path and don't re-normalize the values
                self.index_dct = index_dct

            super(IndexBase, self).__init__()

    def copy(self, **kwargs):
        return super(IndexBase, self).copy(normalize_dtype=False, **kwargs)

    def __repr__(self):
        repr_str = []
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                repr_str.append("=".join([key, str(sorted(val.keys()))]))
            else:
                repr_str.append("=".join([key, str(val)]))
        return "{class_}({attrs})".format(
            class_=type(self).__name__, attrs=", ".join(repr_str)
        )

    @staticmethod
    def normalize_value(dtype, value):
        """
        Normalize value according to index dtype.

        This may apply casts (e.g. integers to floats) or parsing (e.g. timestamps from strings) to the value.

        Parameters
        ----------
        dtype: pyarrow.Type
            Arrow type of the index.
        value: Any
            any value

        Returns
        -------
        value: Any
            normalized value, with a type that matches the index dtype

        Raises
        ------
        ValueError
            If dtype of the index was not set or derived.
        NotImplementedError
            If the dtype cannot be handled.
        """
        if dtype is None:
            raise ValueError(
                "Cannot normalize index values as long as dtype is not set"
            )
        elif pa.types.is_string(dtype):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            else:
                return str(value)
        elif pa.types.is_binary(dtype):
            if isinstance(value, bytes):
                return value
            else:
                return str(value).encode("utf-8")
        elif pa.types.is_date(dtype):
            return pd.Timestamp(value).date()
        elif pa.types.is_temporal(dtype):
            return pd.Timestamp(value).to_datetime64()
        elif pa.types.is_integer(dtype):
            return int(value)
        elif pa.types.is_floating(dtype):
            return float(value)
        elif pa.types.is_boolean(dtype):
            if isinstance(value, str):
                if value.lower() == "false":
                    return False
                elif value.lower() == "true":
                    return True
                else:
                    return ValueError('Cannot parse boolean value "{}"'.format(value))
            return bool(value)
        else:
            raise NotImplementedError(
                "Cannot normalize index value for type {}".format(dtype)
            )

    @property
    def loaded(self):
        """
        Check if the index was already loaded into memory.
        """
        return self.index_dct is not None

    def eval_operator(self, op, value):
        """
        Evaluates a given operator on the index for a given value and returns all
        partition labels allowed by this index.

        Parameters
        ----------
        op: str
            A string representation of the operator to be evaluated. Supported are
            "==", "<=", ">=", "<", ">", "in"
            For details, see documentation of kartothek.serialization
        value: object
            The value to be evaluated
        Returns
        -------
        set: Allowed partition labels
        """
        result = set()
        index_dct = self.index_dct
        index_type = self.dtype
        index_arr = None
        if index_type is not None and index_type and not pa.types.is_date(index_type):
            index_type = index_type.to_pandas_dtype()
            try:
                index_arr = np.fromiter(index_dct.keys(), dtype=index_type)
            except ValueError:
                pass
        if index_arr is None:
            index_arr = np.array(list(index_dct.keys()))

        index = filter_array_like(index_arr, op, value, strict_date_types=True)
        allowed_values = index_arr[index]
        # Need to determine allowed values to include predicates like `in`
        for value in allowed_values:
            result.update(set(self.index_dct[value]))
        return result

    def query(self, value):
        """
        Query this index for a given value. Raises an exception if the index is external
        and not loaded.

        Parameters
        ----------
        value:
            The value that is looked up in the index dictionary.

        Returns
        -------
        keys: [str]
            A list of keys of partitions that contain the corresponding value.
        """
        if self.index_dct is None:
            raise RuntimeError(
                "Index is external. To query an external index, the index ne to be preloaded."
            )

        return self.index_dct.get(IndexBase.normalize_value(self.dtype, value), [])

    def to_dict(self):
        """
        Serialise the object to Python object that can be part of a larger
        dictionary that may be serialised to JSON.

        Returns
        -------
        obj: dict or str
        """
        return self.index_dct

    def update(self, index, inplace=False):
        """

        Returns a new Index object in case of a change.

        The new index object will no longer carry the attribute `index_storage_key`
        since it is no longer a proper representation of the stored index object.

        Parameters
        ----------
        index: [kartothek.core.index.IndexBase]
            The index which should be added to this one
        """
        if not isinstance(index, IndexBase):
            raise TypeError(
                "Need to input an kartothek.core.index.IndexBase object, instead got `{}`".format(
                    type(index)
                )
            )

        # Assume that if one of the indices dtypes is None, they are compatible. In future versions we should make
        # dtype a non-optional parameter
        if self.dtype is not None and index.dtype is not None:
            if self.dtype != index.dtype:
                raise TypeError(
                    "Trying to update an index with different types. Expected `{}` but got `{}`".format(
                        self.dtype, index.dtype
                    )
                )

        if self.column != index.column:
            raise ValueError(
                "Trying to update an index with the wrong column. Got `{}` but expected `{}`".format(
                    index.column, self.column
                )
            )

        if index.index_dct is None or len(index.index_dct) == 0:
            return self
        if inplace:
            new_index_dict = self.index_dct
        else:
            new_index_dict = copy(self.index_dct)

        for value, partition_list in index.index_dct.items():
            old = new_index_dict.get(value, [])
            new_index_dict[value] = list(set(old + partition_list))
        return self.copy(column=self.column, index_dct=new_index_dict, dtype=self.dtype)

    def remove_partitions(self, list_of_partitions, inplace=False):
        """
        Removes a partition from the internal index dictionary

        The new index object will no longer carry the attribute `index_storage_key`
        since it is no longer a proper representation of the stored index object.

        Parameters
        ----------
        list_of_partitions: obj
            The partition to be removed
        inplace: bool, (default: False)
            If `True` the operation is performed inplace and will return the same object
        """
        partitions_to_delete = list_of_partitions
        if not partitions_to_delete:
            return self
        if inplace:
            values_to_remove = []
            for val, partition_list in self.index_dct.items():
                for partition_label in partitions_to_delete:
                    if partition_label in partition_list:
                        partition_list.remove(partition_label)
                    if len(partition_list) == 0:
                        values_to_remove.append(val)
                        break
            for val in values_to_remove:
                del self.index_dct[val]
            # Call the constructor again to reinit the creation timestamp
            return self.copy(
                column=self.column, index_dct=self.index_dct, dtype=self.dtype
            )
        else:
            new_index_dict = {}
            for val, partition_list in self.index_dct.items():
                new_list = set(partition_list) - set(partitions_to_delete)
                if len(new_list) > 0:
                    new_index_dict[val] = list(new_list)
            return self.copy(
                column=self.column, index_dct=new_index_dict, dtype=self.dtype
            )

    def remove_values(self, list_of_values, inplace=False):
        """
        Removes a value from the internal index dictionary

        Parameters
        ----------
        list_of_values: list
            The value to be removed
        inplace: bool, (default: False)
            If `True` the operation is performed inplace and will return the same object
        """
        set_of_values = set(
            IndexBase.normalize_value(self.dtype, v) for v in list_of_values
        )

        if not set_of_values:
            return self

        if inplace:
            for value in set_of_values:
                if value in self.index_dct:
                    del self.index_dct[value]
            # new object for new timestamp, etc
            return self.copy(
                column=self.column,
                index_dct=self.index_dct,
                dtype=self.dtype,
                normalize_dtype=False,
            )
        else:
            new_index_dict = {}
            for val, partition_list in self.index_dct.items():
                if val not in set_of_values:
                    new_index_dict[val] = partition_list

            return self.copy(
                column=self.column,
                index_dct=new_index_dict,
                dtype=self.dtype,
                normalize_dtype=False,
            )

    def __eq__(self, other):
        if not isinstance(other, IndexBase):
            return False
        if self.column != other.column:
            return False
        if (self.dtype is None) and (other.dtype is not None):
            return False
        if (self.dtype is not None) and (other.dtype is None):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self.index_dct) != len(other.index_dct):
            return False
        for col, partition_list in self.index_dct.items():
            if col not in other.index_dct:
                return False
            if set(partition_list) != set(other.index_dct[col]):
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def as_flat_series(self, compact=False, partitions_as_index=False):
        """
        Convert the Index object to a pandas.Series

        Parameters
        ----------
        compact: bool, optional
            If True, the index will be unique and the Series.values will be a list of partitions/values
        partitions_as_index: bool, optional
            If True, the relation between index values and partitions will be reverted for the output
        """
        table = _index_dct_to_table(self.index_dct, column=self.column)
        df = table.to_pandas(date_as_object=True)
        result_column = _PARTITION_COLUMN_NAME
        # This is the way the dictionary is directly translated
        # value: [partition]
        if compact and not partitions_as_index:
            return df.set_index(self.column)[result_column]

        # In all other circumstances we need a flat series first
        # value: part_1
        # value: part_2
        # value2: part_1
        if partitions_as_index or not compact:
            keys = np.concatenate(df[_PARTITION_COLUMN_NAME].values)

            lengths = df[_PARTITION_COLUMN_NAME].apply(len).values
            values_index = np.repeat(np.arange(len(df)), lengths)
            values = df[self.column].values[values_index]

            df = pd.DataFrame({_PARTITION_COLUMN_NAME: keys, self.column: values})

        # if it is not inverted and not compact, we're done
        if partitions_as_index:
            result_index = _PARTITION_COLUMN_NAME
            if compact:
                df = df.groupby(df[result_index]).apply(
                    lambda x: x[self.column].tolist()
                )
                df.name = self.column
            else:
                df = df.set_index(result_index)[self.column]
        else:
            df = df.set_index(self.column)[_PARTITION_COLUMN_NAME]
        return df


class PartitionIndex(IndexBase):
    """
    An Index class representing partition indices (sometimes also referred to as primary indices).
    A PartitionIndex is usually constructed by parsing the partition filenames which encode index information.

    The constructor for this class should usually not be called explicitly but indices should be created by e.g.
    :meth:`kartothek.core.dataset.DatasetMetadataBase.load_partition_indices`
    """

    def __init__(self, column, index_dct=None, dtype=None, normalize_dtype=True):
        if dtype is None:
            raise ValueError(
                'PartitionIndex dtype of column "{}" cannot be None!'.format(column)
            )
        super(PartitionIndex, self).__init__(
            column=column,
            index_dct=index_dct,
            dtype=dtype,
            normalize_dtype=normalize_dtype,
        )

    def __eq__(self, other):
        if not isinstance(other, PartitionIndex):
            return False
        return super(PartitionIndex, self).__eq__(other)


class ExplicitSecondaryIndex(IndexBase):
    """
    An Index class representing an explicit, secondary index which is calculated and stored next to the dataset.
    In contrast to the `PartitionIndex` this needs to be calculated by an explicit pass over the data. All mutations of
    this class will erase the reference to the physical file and the storage of the mutated object will write to a new
    storage key.
    """

    def __init__(
        self,
        column,
        index_dct=None,
        index_storage_key=None,
        dtype=None,
        normalize_dtype=True,
    ):
        if (index_dct is None) and not index_storage_key:
            raise ValueError("No valid index source specified")
        self.index_storage_key = index_storage_key
        super(ExplicitSecondaryIndex, self).__init__(
            column=column,
            index_dct=index_dct,
            dtype=dtype,
            normalize_dtype=normalize_dtype,
        )

    def copy(self, **kwargs):
        if kwargs:
            index_storage_key = None
        else:
            index_storage_key = self.index_storage_key
        return super(IndexBase, self).copy(
            index_storage_key=index_storage_key, **kwargs
        )

    def __eq__(self, other):
        if not isinstance(other, ExplicitSecondaryIndex):
            return False
        if self.index_storage_key != other.index_storage_key:
            return False
        try:
            return super(ExplicitSecondaryIndex, self).__eq__(other)
        except TypeError:  # an `index_dct == None`
            if self.index_dct != other.index_dct:
                return False
            return True

    @staticmethod
    def from_v2(column, dct_or_str):
        """
        Create an index instance from a version 2 Python structure.

        Parameters
        ----------
        column: str
            Name of the column this index provides lookup for
        dct_or_str: dict or str
            Either the storage key of the external index or the index itself
            as a Python object structure.

        Returns
        -------
        index: [kartothek.core.index.ExplicitSecondaryIndex]
        """
        if isinstance(dct_or_str, str):
            # External index
            return ExplicitSecondaryIndex(column=column, index_storage_key=dct_or_str)
        else:
            return ExplicitSecondaryIndex(column=column, index_dct=dct_or_str)

    def to_dict(self):
        """
        Serialise the object to Python object that can be part of a larger
        dictionary that may be serialised to JSON.

        Returns
        -------
        obj: dict or str
        """
        if self.index_dct is None:
            # FIXME: This should not return a simple string
            return self.index_storage_key
        else:
            return self.index_dct

    def store(self, store, dataset_uuid):
        """
        Store the index as a parquet file

        If compatible, the new keyname will be the name stored under the attribute `index_storage_key`.
        If this attribute is None, a new key will be generated of the format

            `{dataset_uuid}/indices/{column}/{timestamp}.by-dataset-index.parquet`

        where the timestamp is in nanosecond accuracy and is created upon Index object initialization

        Parameters
        ----------
        store: object
        dataset_uuid: str
        """
        storage_key = None

        if (
            self.index_storage_key is not None
            and dataset_uuid
            and dataset_uuid in self.index_storage_key
        ):
            storage_key = self.index_storage_key
        if storage_key is None:
            storage_key = "{dataset_uuid}/indices/{column}/{timestamp}{suffix}".format(
                dataset_uuid=dataset_uuid,
                suffix=naming.EXTERNAL_INDEX_SUFFIX,
                column=quote(self.column),
                timestamp=quote(self.creation_time.isoformat()),
            )

        table = _index_dct_to_table(self.index_dct, self.column)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)

        store.put(storage_key, buf.getvalue().to_pybytes())
        return storage_key

    def load(self, store):
        """
        Load an external index into memory. Returns a new index object that
        contains the index dictionary. Returns itself if the index is internal
        or an already loaded index.

        Parameters
        ----------
        store: Object
            Object that implements the .get method for file/object loading.

        Returns
        -------
        index: [kartothek.core.index.ExplicitSecondaryIndex]
        """
        if self.index_dct is None:
            index_buffer = store.get(self.index_storage_key)
            index_dct, column_type = _parquet_bytes_to_dict(self.column, index_buffer)

            return ExplicitSecondaryIndex(
                column=self.column,
                index_dct=index_dct,
                dtype=column_type,
                index_storage_key=self.index_storage_key,
                normalize_dtype=False,
            )
        else:
            return self

    def __getstate__(self):
        if self.index_dct is None:
            return (self.column, self.index_storage_key, self.dtype, None)

        table = _index_dct_to_table(self.index_dct, self.column)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        parquet_bytes = buf.getvalue().to_pybytes()
        # Since `self.dtype` will be inferred by parquet bytes, do not return
        # this argument during serialization to avoid unnecessary memory consumption
        return (self.column, self.index_storage_key, None, parquet_bytes)

    def __setstate__(self, state):
        column, storage_key, column_type, parquet_bytes = state
        if parquet_bytes is None:
            index_dct = None
        else:
            index_dct, column_type = _parquet_bytes_to_dict(column, parquet_bytes)
        self.__init__(
            column=column,
            index_dct=index_dct,
            index_storage_key=storage_key,
            dtype=column_type,
            normalize_dtype=False,
        )


def merge_indices(list_of_indices):
    """
    Merge a list of index dictionaries

    Parameters
    ----------
    list_of_indices: list of tuple
        A list of tuples holding index information

        Format: [ (partition_label, index_dict) ]
    """
    final_indices = {}
    if not list_of_indices:
        return final_indices

    # shortcut can only be applied to Index object since the dict path needs the label in the passed tuple
    if len(list_of_indices) == 1:
        return list_of_indices[0]
    elif len(list_of_indices) > 2:
        first = merge_indices(list_of_indices[0::2])
        second = merge_indices(list_of_indices[1::2])
        return merge_indices([first, second])

    for indices in list_of_indices:
        for column, index in indices.items():
            if column in final_indices:
                final_indices[column] = final_indices[column].update(index)
            else:
                final_indices[column] = index

    return final_indices


def remove_partitions_from_indices(index_dict, partitions):
    """
    Remove a given list of partitions from a kartothek index dictionary

    Parameters
    ----------
    index_dict: dict of Index
        A dictionary holding kartothek indices
    partitions: list
        A list of partition labels which should be removed form the index objects
    """
    new_index_dict = {}
    for column, index in index_dict.items():
        new_index_dict[column] = index.remove_partitions(partitions)
    return new_index_dict


def filter_indices(index_dict, partitions):
    """
    Filter a kartothek index dictionary such that only the provided list of partitions is included
    in the index dictionary

    All indices must be embedded!

    Parameters
    ----------
    index_dict: dict of Index
        A dictionary holding kartothek indices
    partition_list: list
        A list of partition labels which are allowed in the output dictionary
    """
    new_index_dict = {}
    index_types = {}
    types = {}
    for column, index in index_dict.items():
        new_index_dict[column] = {}
        types[column] = index.dtype
        index_types[column] = type(index)
        for index_value, partition_list in index.to_dict().items():
            new_index_dict[column][index_value] = [
                part_label for part_label in partition_list if part_label in partitions
            ]

    for column, index_dict in new_index_dict.items():
        new_index_dict[column] = index_types[column](
            column=column, index_dct=index_dict, dtype=types[column]
        )
    return new_index_dict


def _parquet_bytes_to_dict(column, index_buffer):
    reader = pa.BufferReader(index_buffer)
    # This can be done much more efficient but would take a lot more
    # time to implement so this will be only done on request.
    table = pq.read_table(reader)
    column_type = table.schema.field_by_name(column).type

    # `datetime.datetime` objects have a precision of up to microseconds only, so arrow
    # parses the type to `pa.timestamp("us")`. Since the
    # values are normalized to `numpy.datetime64[ns]` anyways, we do not care about this
    # and load the column type as `pa.timestamp("ns")`
    if column_type == pa.timestamp("us"):
        column_type = pa.timestamp("ns")

    df = _fix_pyarrow_07992_table(table).to_pandas()  # Could eventually be phased out

    index_dct = dict(
        zip(df[column].values, (list(x) for x in df[_PARTITION_COLUMN_NAME].values))
    )
    return index_dct, column_type


def _index_dct_to_table(index_dct, column):
    keys = index_dct.keys()
    keys_type = None
    if len(keys) > 0:
        probe = next(iter(keys))
        if isinstance(probe, np.datetime64):
            keys_type = pa.timestamp(
                "ns"
            )  # workaround pyarrow type inference bug (ARROW-2554)
        elif isinstance(probe, pd.Timestamp):
            keys = [d.to_datetime64() for d in keys]
            keys_type = pa.timestamp(
                "ns"
            )  # workaround pyarrow type inference bug (ARROW-2554)
        elif isinstance(probe, np.bool_):
            keys = [bool(d) for d in keys]

    # This is because of ARROW-1646:
    #   [Python] pyarrow.array cannot handle NumPy scalar types
    # Additional note: pyarrow.array is supposed to infer type automatically.
    # But the inferred type is not enough to hold np.uint64. Until this is fixed in
    # upstream Arrow, we have to retain the following line
    keys = np.array(list(keys))

    labeled_array = pa.array(keys, type=keys_type)
    partition_array = pa.array(list(index_dct.values()))

    return pa.Table.from_arrays(
        [labeled_array, partition_array], names=[column, _PARTITION_COLUMN_NAME]
    )
