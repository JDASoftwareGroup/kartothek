# -*- coding: utf-8 -*-


import pickle
from itertools import permutations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import simplejson
from dask.dataframe.utils import make_meta as dask_make_meta
from packaging.version import parse as parse_version

from kartothek.core.common_metadata import (
    SchemaWrapper,
    _diff_schemas,
    _get_common_metadata_key,
    empty_dataframe_from_schema,
    make_meta,
    read_schema_metadata,
    store_schema_metadata,
    validate_compatible,
    validate_shared_columns,
)
from kartothek.serialization import ParquetSerializer

try:
    arrow_version = parse_version(pa.__version__)
    ARROW_DEV = arrow_version.is_devrelease
    del arrow_version
except Exception:
    ARROW_DEV = True


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
    expected_schema = pa.schema(fields)

    assert actual_schema.remove_metadata() == expected_schema


def test_schema_roundtrip(df_all_types, store):
    expected_meta = make_meta(df_all_types, origin="df_all_types")
    store_schema_metadata(
        expected_meta, dataset_uuid="dataset_uuid", store=store, table="table"
    )
    result = read_schema_metadata(
        dataset_uuid="dataset_uuid", store=store, table="table"
    )
    assert result == expected_meta


def test_pickle(df_all_types):
    obj1 = make_meta(df_all_types, origin="df_all_types")
    s = pickle.dumps(obj1)
    obj2 = pickle.loads(s)
    assert obj1 == obj2


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


def test_unicode_col():
    df = pd.DataFrame({"fö": [1]})
    make_meta(df, origin="df")


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


def test_reorder(df_all_types):
    df2 = df_all_types.copy()
    df2 = df2.reindex(reversed(df_all_types.columns), axis=1)
    expected = make_meta(df_all_types, origin="df_all_types")
    actual = make_meta(df2, origin="df2")
    assert expected == actual


def test_reorder_partition_keys(df_all_types):
    partition_keys = ["int8", "uint8", "array_unicode"]
    df2 = df_all_types.copy()
    df2 = df2.reindex(reversed(df_all_types.columns), axis=1)

    expected = make_meta(
        df_all_types, origin="df_all_types", partition_keys=partition_keys
    )
    assert expected.names[: len(partition_keys)] == partition_keys

    actual = make_meta(df2, origin="df2", partition_keys=partition_keys)
    assert expected == actual


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


def test_validate_compatible_same(df_all_types):
    schema1 = make_meta(df_all_types, origin="1")
    schema2 = make_meta(df_all_types, origin="2")
    schema3 = make_meta(df_all_types, origin="3")
    validate_compatible([])
    validate_compatible([schema1])
    validate_compatible([schema1, schema2])
    validate_compatible([schema1, schema2, schema3])


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


def test_validate_compatible_different(df_all_types):
    df2 = df_all_types.loc[:, df_all_types.columns[:2]].copy()
    schema1 = make_meta(df_all_types, origin="1")
    schema2 = make_meta(df2, origin="2")
    with pytest.raises(ValueError) as exc:
        validate_compatible([schema1, schema2])
    assert str(exc.value).startswith("Schema violation")


def test_validate_shared_columns_same(df_all_types):
    schema1 = make_meta(df_all_types, origin="1")
    schema2 = make_meta(df_all_types, origin="2")
    schema3 = make_meta(df_all_types, origin="3").remove_metadata()
    validate_shared_columns([])
    validate_shared_columns([schema1])
    validate_shared_columns([schema1, schema2])
    with pytest.raises(ValueError):
        validate_shared_columns([schema1, schema2, schema3])
    validate_shared_columns([schema1, schema2, schema3], ignore_pandas=True)
    validate_shared_columns(
        [schema1.remove_metadata(), schema2.remove_metadata(), schema3]
    )


def test_validate_shared_columns_null_value(df_all_types):
    schema1 = make_meta(df_all_types, origin="1")
    schema2 = make_meta(df_all_types.drop(0), origin="2")
    schema3 = make_meta(df_all_types, origin="3").remove_metadata()
    validate_shared_columns([])
    validate_shared_columns([schema1])
    validate_shared_columns([schema1, schema2])
    with pytest.raises(ValueError):
        validate_shared_columns([schema1, schema2, schema3])
    validate_shared_columns([schema1, schema2, schema3], ignore_pandas=True)
    validate_shared_columns(
        [schema1.remove_metadata(), schema2.remove_metadata(), schema3]
    )


def test_validate_shared_columns_no_share(df_all_types):
    schema1 = make_meta(df_all_types.loc[:, df_all_types.columns[0:2]], origin="1")
    schema2 = make_meta(df_all_types.loc[:, df_all_types.columns[2:4]], origin="2")
    schema3 = make_meta(df_all_types.loc[:, df_all_types.columns[4:6]], origin="3")
    validate_shared_columns([])
    validate_shared_columns([schema1])
    validate_shared_columns([schema1, schema2])
    validate_shared_columns([schema1, schema2, schema3])


@pytest.mark.parametrize("remove_metadata", [True, False])
def test_validate_shared_columns_fail(df_all_types, remove_metadata):
    df2 = df_all_types.copy()
    df2["uint16"] = df2["uint16"].astype(float)
    schema1 = make_meta(df_all_types, origin="1")
    schema2 = make_meta(df2, origin="2")
    if remove_metadata:
        schema1 = schema1.remove_metadata()
        schema2 = schema2.remove_metadata()
    with pytest.raises(ValueError) as exc:
        validate_shared_columns([schema1, schema2])
    assert str(exc.value).startswith('Found incompatible entries for column "uint16"')


def test_validate_empty_dataframe(
    df_all_types, df_all_types_schema, df_all_types_empty_schema
):
    # Do not raise in case one of the schemas is of an empty dataframe
    # Test all permutations to avoid that the implementation is sensitive on whether
    # the first schema is empty/non-empty
    for schemas in permutations([df_all_types_schema, df_all_types_empty_schema]):
        validate_compatible(schemas)
    validate_compatible([df_all_types_empty_schema, df_all_types_empty_schema])


@pytest.mark.parametrize(
    "corrupt_column,corrupt_value,corrupt_dtype",
    [
        # reference column is a native type
        ("int8", -1.1, np.float64),
        ("int8", "a", np.object),
        # reference column is an object
        ("unicode", -1.1, np.float64),
        ("unicode", 1, np.int64),
        pytest.param(
            "unicode",
            None,
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This results in a `null` column which cannot be compared and must not fail",
            ),
        ),
        # reference column is a native typed array
        ("array_int64", np.array([1.0], dtype=np.float64), np.object),
        ("array_int64", np.array(["a"], dtype=np.object), np.object),
        # reference column is an object types arrayarray
        ("array_unicode", np.array([1], dtype=np.int8), np.object),
        ("array_unicode", np.array([1.0], dtype=np.float64), np.object),
    ],
)
def test_validate_empty_dataframe_corrupt_raises(
    df_all_types,
    df_all_types_schema,
    df_all_types_empty_schema,
    corrupt_column,
    corrupt_value,
    corrupt_dtype,
):
    # In case there is something wrong with the schema, raise!

    # First, an integer column carries a float or an object.
    df_corrupt = df_all_types.copy()
    # for value, dtype in [(-1.1, np.float64), ('a', np.object)]:
    df_corrupt[corrupt_column] = pd.Series([corrupt_value], dtype=corrupt_dtype)
    df_corrupt_meta = make_meta(df_corrupt, origin="1")
    # Raise when comparing the proper to the corrupt schema
    for schemas in permutations([df_all_types_schema, df_corrupt_meta]):
        with pytest.raises(ValueError):
            validate_compatible(schemas)
    # Also raise if there is a schema originating from an empty DF to make
    # sure the emptiness doesn't cancel the validation
    for schemas in permutations(
        [df_all_types_schema, df_corrupt_meta, df_all_types_empty_schema]
    ):
        with pytest.raises(ValueError):
            validate_compatible(schemas)


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


@pytest.mark.parametrize("index", [pd.Int64Index([0]), pd.RangeIndex(start=0, stop=1)])
def test_schema_dataframe_rountrip(index, df_all_types):
    df = pd.DataFrame(df_all_types, index=index)

    schema = make_meta(df, origin="1")
    actual_df = empty_dataframe_from_schema(schema, date_as_object=True)
    validate_compatible([schema, make_meta(actual_df, origin="2")])


def test_empty_dataframe_from_schema(df_all_types):
    schema = make_meta(df_all_types, origin="1")
    actual_df = empty_dataframe_from_schema(schema)

    expected_df = df_all_types.loc[[]]
    expected_df["date"] = pd.Series([], dtype="datetime64[ns]")
    for c in expected_df.columns:
        if c.startswith("float"):
            expected_df[c] = pd.Series([], dtype=float)
        if c.startswith("int"):
            expected_df[c] = pd.Series([], dtype=int)
        if c.startswith("uint"):
            expected_df[c] = pd.Series([], dtype=np.uint64)

    pdt.assert_frame_equal(actual_df, expected_df)


def test_empty_dataframe_from_schema_columns(df_all_types):
    schema = make_meta(df_all_types, origin="1")
    actual_df = empty_dataframe_from_schema(schema, ["uint64", "int64"])

    expected_df = df_all_types.loc[[], ["uint64", "int64"]]
    pdt.assert_frame_equal(actual_df, expected_df)


@pytest.mark.xfail(ARROW_DEV, reason="Format not stbale")
def test_diff_schemas(df_all_types):
    # Prepare a schema with one missing, one additional and one changed column
    df2 = df_all_types.drop(columns=df_all_types.columns[0])
    df2["new_col"] = pd.Series(df_all_types["bool"])
    df2["int16"] = df2["int16"].astype(float)
    df2 = df2.reset_index(drop=True)

    schema2 = make_meta(df2, origin="2")
    schema1 = make_meta(df_all_types, origin="1")
    diff = _diff_schemas(schema1, schema2)
    expected_arrow_diff = """Arrow schema:
@@ -1,5 +1,3 @@

-array_float32: list<item: double>
-  child 0, item: double
 array_float64: list<item: double>
   child 0, item: double
 array_int16: list<item: int64>
@@ -26,10 +24,11 @@

 datetime64: timestamp[ns]
 float32: double
 float64: double
-int16: int64
+int16: double
 int32: int64
 int64: int64
 int8: int64
+new_col: bool
 null: null
 uint16: uint64
 uint32: uint64

"""
    expected_pandas_diff = """Pandas_metadata:
@@ -3,12 +3,7 @@

                      'name': None,
                      'numpy_type': 'object',
                      'pandas_type': 'unicode'}],
- 'columns': [{'field_name': 'array_float32',
-              'metadata': None,
-              'name': 'array_float32',
-              'numpy_type': 'object',
-              'pandas_type': 'list[float64]'},
-             {'field_name': 'array_float64',
+ 'columns': [{'field_name': 'array_float64',
               'metadata': None,
               'name': 'array_float64',
               'numpy_type': 'object',
@@ -91,8 +86,8 @@

              {'field_name': 'int16',
               'metadata': None,
               'name': 'int16',
-              'numpy_type': 'int64',
-              'pandas_type': 'int64'},
+              'numpy_type': 'float64',
+              'pandas_type': 'float64'},
              {'field_name': 'int32',
               'metadata': None,
               'name': 'int32',
@@ -108,6 +103,11 @@

               'name': 'int8',
               'numpy_type': 'int64',
               'pandas_type': 'int64'},
+             {'field_name': 'new_col',
+              'metadata': None,
+              'name': 'new_col',
+              'numpy_type': 'bool',
+              'pandas_type': 'bool'},
              {'field_name': 'null',
               'metadata': None,
               'name': 'null',"""

    assert diff == expected_arrow_diff + expected_pandas_diff


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


def test_make_meta_column_normalization_pyarrow_schema():
    # GH228
    df = pd.DataFrame(
        [{"part": 1, "id": 1, "col1": "abc"}, {"part": 2, "id": 2, "col1": np.nan}],
        # Kartothek normalizes field order s.t. partition keys are first and the
        # rest is alphabetically. This is reverse.
        columns=["col1", "id", "part"],
    )
    schema = make_meta(
        pa.Schema.from_pandas(df), origin="gh228", partition_keys=["part"]
    )
    fields = [
        pa.field("part", pa.int64()),
        pa.field("col1", pa.string()),
        pa.field("id", pa.int64()),
    ]
    expected_schema = pa.schema(fields)

    assert schema.internal().equals(expected_schema, check_metadata=False)
