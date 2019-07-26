# -*- coding: utf-8 -*-


from itertools import permutations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import simplejson
from dask.dataframe.utils import make_meta as dask_make_meta

from kartothek.core._compat import ARROW_LARGER_EQ_0130
from kartothek.core.common_metadata import (
    SchemaWrapper,
    _get_common_metadata_key,
    make_meta,
    read_schema_metadata,
    store_schema_metadata,
    validate_compatible,
)
from kartothek.serialization import ParquetSerializer


def test_store_schema_metadata(store, df_all_types):
    store_schema_metadata(
        schema=make_meta(df_all_types, origin="df_all_types"),
        dataset_uuid="some_uuid",
        store=store,
        table="some_table",
    )

    key = "some_uuid/some_table/_common_metadata"
    assert key in store.keys()
    pq_file = pq.ParquetFile(store.open(key))
    actual_schema = pq_file.schema.to_arrow_schema()
    fields = [
        pa.field("array_float32", pa.list_(pa.float64())),
        pa.field("array_float64", pa.list_(pa.float64())),
        pa.field("array_int16", pa.list_(pa.int64())),
        pa.field("array_int32", pa.list_(pa.int64())),
        pa.field("array_int64", pa.list_(pa.int64())),
        pa.field("array_int8", pa.list_(pa.int64())),
        pa.field("array_uint16", pa.list_(pa.uint64())),
        pa.field("array_uint32", pa.list_(pa.uint64())),
        pa.field("array_uint64", pa.list_(pa.uint64())),
        pa.field("array_uint8", pa.list_(pa.uint64())),
        pa.field("array_unicode", pa.list_(pa.string())),
        pa.field("bool", pa.bool_()),
        pa.field("byte", pa.binary()),
        pa.field("date", pa.date32()),
        pa.field("datetime64", pa.timestamp("us")),
        pa.field("float32", pa.float64()),
        pa.field("float64", pa.float64()),
        pa.field("int16", pa.int64()),
        pa.field("int32", pa.int64()),
        pa.field("int64", pa.int64()),
        pa.field("int8", pa.int64()),
        pa.field("null", pa.null()),
        pa.field("uint16", pa.uint64()),
        pa.field("uint32", pa.uint64()),
        pa.field("uint64", pa.uint64()),
        pa.field("uint8", pa.uint64()),
        pa.field("unicode", pa.string()),
    ]
    if not ARROW_LARGER_EQ_0130:
        fields.append(pa.field("__index_level_0__", pa.int64()))
    expected_schema = pa.schema(fields)

    assert actual_schema.remove_metadata() == expected_schema


def test_wrapper(df_all_types):
    obj = make_meta(df_all_types, origin="df_all_types")
    assert isinstance(obj, SchemaWrapper)
    assert isinstance(obj.metadata, dict)
    assert isinstance(obj.internal(), pa.Schema)

    obj2 = make_meta(df_all_types, origin="df_all_types")
    assert obj == obj2
    assert obj == obj2.internal()
    assert obj.equals(obj2)
    assert obj.equals(obj2.internal())
    assert repr(obj) == repr(obj.internal())

    assert isinstance(obj[0], pa.Field)


def test_strip_categories():
    input_df = pd.DataFrame(
        {"categories": pd.Series(["a", "b", "c", "a"], dtype="category")}
    )

    assert len(input_df["categories"].cat.categories) > 1
    meta = make_meta(input_df, origin="input_df")
    # We strip categories to have the underlying type as the information.
    # Categories also include their categorical values as type information,
    # we're not interested in keeping them the same in all partitions.
    assert not pa.types.is_dictionary(meta[0].type)


def test_compat_old_rw_path(df_all_types, store):
    # strip down DF before some column types weren't supported before anyway
    df = df_all_types[
        [
            c
            for c in df_all_types.columns
            if (
                not c.startswith("array_")  # array types (always null)
                and c != "unicode"  # unicode type (alway null)
                and "8" not in c  # 8 bit types are casted to 64 bit
                and "16" not in c  # 16 bit types are casted to 64 bit
                and "32" not in c  # 32 bit types are casted to 64 bit
            )
        ]
    ]
    expected_meta = make_meta(df, origin="df")

    # old schema write path
    old_meta = dask_make_meta(df)
    pa_table = pa.Table.from_pandas(old_meta)
    buf = pa.BufferOutputStream()
    pq.write_table(pa_table, buf, version="2.0")
    key_old = _get_common_metadata_key("dataset_uuid_old", "table")
    store.put(key_old, buf.getvalue().to_pybytes())

    actual_meta = read_schema_metadata(
        dataset_uuid="dataset_uuid_old", store=store, table="table"
    )
    validate_compatible([actual_meta, expected_meta])

    store_schema_metadata(
        schema=make_meta(df, origin="df"),
        dataset_uuid="dataset_uuid_new",
        store=store,
        table="table",
    )
    key_new = _get_common_metadata_key("dataset_uuid_new", "table")
    actual_df = ParquetSerializer.restore_dataframe(key=key_new, store=store)
    actual_df["date"] = actual_df["date"].dt.date
    pdt.assert_frame_equal(actual_df, old_meta)


@pytest.mark.parametrize("remove_metadata", [True, False])
@pytest.mark.parametrize("ignore_pandas", [True, False])
def test_validate_compatible_other_pandas(df_all_types, remove_metadata, ignore_pandas):
    def _with_pandas(version):
        schema = make_meta(df_all_types, origin=version)
        metadata = schema.metadata
        pandas_metadata = simplejson.loads(metadata[b"pandas"].decode("utf8"))
        pandas_metadata["pandas_version"] = version
        metadata[b"pandas"] = simplejson.dumps(pandas_metadata).encode("utf8")
        schema = SchemaWrapper(pa.schema(schema, metadata), version)
        if remove_metadata:
            return schema.remove_metadata()
        else:
            return schema

    schema1 = make_meta(df_all_types, origin="all")
    schema2 = _with_pandas("0.19.0")
    schema3 = _with_pandas("0.99.0")
    if remove_metadata and not ignore_pandas:
        # This should fail as long as we have the metadata attached
        with pytest.raises(ValueError):
            validate_compatible(
                [schema1, schema2, schema3], ignore_pandas=ignore_pandas
            )
        schema1 = schema1.remove_metadata()
    validate_compatible([schema1, schema2, schema3], ignore_pandas=ignore_pandas)


def test_validate_different_cats_same_type():
    input_df = pd.DataFrame(
        {"categories": pd.Series(["a", "b", "c", "a"], dtype="category")}
    )
    input_df_2 = pd.DataFrame(
        {"categories": pd.Series(["f", "e", "e", "f"], dtype="category")}
    )
    input_df_3 = pd.DataFrame({"categories": pd.Series(["f", "e", "e", "f"])})

    meta = make_meta(input_df, origin="1")
    meta_2 = make_meta(input_df_2, origin="2")
    meta_3 = make_meta(input_df_3, origin="3")
    validate_compatible([meta, meta_2, meta_3])


def test_validate_different_cats_different_type():
    input_df = pd.DataFrame(
        {"categories": pd.Series(["a", "b", "c", "a"], dtype="category")}
    )
    input_df_2 = pd.DataFrame(
        {"categories": pd.Series([b"f", b"e", b"e", b"f"], dtype="category")}
    )

    meta = make_meta(input_df, origin="1")
    meta_2 = make_meta(input_df_2, origin="2")
    with pytest.raises(ValueError):
        validate_compatible([meta, meta_2])


def test_validate_schema_non_overlapping_nulls(df_all_types_schema):
    """
    Test that two schemas with non-overlapping null columns are valid
    """
    first_ix = np.random.randint(len(df_all_types_schema))
    second_ix = first_ix
    while second_ix == first_ix:
        second_ix = np.random.randint(len(df_all_types_schema))

    first_null = pa.field(name=df_all_types_schema.names[first_ix], type=pa.null())
    first_schema = df_all_types_schema.set(first_ix, first_null)

    second_null = pa.field(name=df_all_types_schema.names[second_ix], type=pa.null())
    second_schema = df_all_types_schema.set(second_ix, second_null)

    for schemas in permutations([first_schema, second_schema]):
        reference_schema = validate_compatible(schemas)

        # The reference schema should be the original schema
        # with the columns reconstructed
        assert df_all_types_schema == reference_schema
