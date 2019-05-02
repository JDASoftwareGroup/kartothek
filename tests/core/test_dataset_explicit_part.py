# -*- coding: utf-8 -*-


import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import simplejson

import kartothek.core._zmsgpack as msgpack
from kartothek.core.common_metadata import make_meta, store_schema_metadata
from kartothek.core.dataset import DatasetMetadata
from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
from kartothek.core.testing import cm_frozen_time

# Basic functionality tests.
#
# In test_reference_data.py we test even more constellations of datasets.

TIME_TO_FREEZE = datetime.datetime(2000, 1, 1, 1, 1, 1, 1)


@pytest.fixture
def frozen_time():
    with cm_frozen_time(TIME_TO_FREEZE):
        yield


def test_roundtrip_empty(metadata_version):
    ds = DatasetMetadata(uuid="dataset_uuid", metadata_version=metadata_version)
    assert ds == ds.from_dict(ds.to_dict())


def test_roundtrip_empty_with_store(store, metadata_version):
    dataset_uuid = "dataset_uuid"
    dataset = DatasetMetadata(uuid=dataset_uuid, metadata_version=metadata_version)
    store.put(
        "{}.by-dataset-metadata.json".format(dataset_uuid),
        simplejson.dumps(dataset.to_dict()).encode("utf-8"),
    )
    assert dataset == DatasetMetadata.load_from_store(dataset_uuid, store)


def test_roundtrip(metadata_version, frozen_time):
    expected = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid",
        "metadata": {"key": "value", "creation_time": "2000-01-01 01:01:01"},
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
        "partition_keys": [],
        "indices": {"p_id": {"1": ["part_1"]}},
    }
    result = DatasetMetadata.from_dict(expected).to_dict()
    assert expected == result


def test_roundtrip_no_metadata(metadata_version, frozen_time):
    expected = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid",
        "metadata": {"creation_time": "2000-01-01 01:01:01"},
        "partition_keys": [],
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
    }
    result = DatasetMetadata.from_dict(expected).to_dict()
    assert expected == result


def test_roundtrip_json(metadata_version):
    expected = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid",
        "metadata": {"key": "value", "creation_time": "2000-01-01 01:01:01"},
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
        "partition_keys": [],
        "indices": {"p_id": {"1": ["part_1"]}},
    }

    result = simplejson.loads(
        DatasetMetadata.from_buffer(simplejson.dumps(expected)).to_json()
    )
    assert expected == result


def test_roundtrip_msgpack():
    expected = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "uuid",
        "metadata": {"key": "value", "creation_time": "2000-01-01 01:01:01"},
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
        "partition_keys": [],
        "indices": {"p_id": {"1": ["part_1"]}},
    }

    result = msgpack.unpackb(
        DatasetMetadata.from_buffer(
            msgpack.packb(expected), format="msgpack"
        ).to_msgpack()
    )
    assert expected == result


def test_invalid_uuid():
    expected = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "uuid.",
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
    }
    with pytest.raises(ValueError):
        DatasetMetadata.from_dict(expected)

    expected = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "mañana",
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
    }
    with pytest.raises(ValueError):
        DatasetMetadata.from_dict(expected)


def test_complicated_uuid():
    expected = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "uuid+namespace-attribute12_underscored",
        "partitions": {"part_1": {"files": {"core": "file.parquet"}}},
    }
    DatasetMetadata.from_dict(expected)


def test_read_table_meta(store):
    meta_dct = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "dataset_uuid",
        "partitions": {
            "location_id=1/part_1": {
                "files": {
                    "table1": "dataset_uuid/table1/location_id=1/part_1.parquet",
                    "table2": "dataset_uuid/table2/location_id=1/part_1.parquet",
                }
            }
        },
    }
    df1 = pd.DataFrame(
        {"location_id": pd.Series([1], dtype=int), "x": pd.Series([True], dtype=bool)}
    )
    df2 = pd.DataFrame(
        {"location_id": pd.Series([1], dtype=int), "y": pd.Series([1.0], dtype=float)}
    )
    schema1 = make_meta(df1, origin="1")
    schema2 = make_meta(df2, origin="2")
    store_schema_metadata(schema1, "dataset_uuid", store, "table1")
    store_schema_metadata(schema2, "dataset_uuid", store, "table2")

    dmd = DatasetMetadata.load_from_dict(meta_dct, store)

    actual = dmd.table_meta
    expected = {"table1": schema1, "table2": schema2}
    assert actual == expected


def test_load_indices_embedded(metadata_version):
    expected = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid+namespace-attribute12_underscored",
        "partitions": {"part_1": {"files": {"core_data": "file.parquest"}}},
        "indices": {
            "product_id": {
                "1": ["part_1"],
                "2": ["part_1"],
                "100": ["part_1"],
                "34": ["part_1"],
            }
        },
    }
    dmd = DatasetMetadata.from_dict(expected)
    assert "product_id" in dmd.indices

    with pytest.raises(KeyError):
        dmd.load_index("not there", store=None)

    dmd_loaded = dmd.load_index("product_id", store=None)
    assert "product_id" in dmd_loaded.indices


def test_load_all_indices(store, metadata_version):
    meta_dct = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid+namespace-attribute12_underscored",
        "partitions": {
            "location_id=1/part_1": {
                "files": {
                    "core_data": "dataset_uuid/table/location_id=1/part_1.parquet"
                }
            }
        },
        "indices": {
            "product_id": {
                "1": ["part_1"],
                "2": ["part_1"],
                "100": ["part_1"],
                "34": ["part_1"],
            }
        },
    }
    dmd = DatasetMetadata.from_dict(meta_dct)
    dmd.table_meta["core_data"] = make_meta(
        pd.DataFrame({"location_id": pd.Series([1], dtype=int)}), origin="core"
    )

    dmd = dmd.load_all_indices(store)

    assert "product_id" in dmd.indices
    assert isinstance(dmd.indices["product_id"], ExplicitSecondaryIndex)

    assert "location_id" in dmd.indices
    assert isinstance(dmd.indices["location_id"], PartitionIndex)

    assert len(dmd.indices) == 2


def test_load_from_store_with_indices(store):
    meta_dct = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "uuid",
        "partitions": {
            "product_id=1/part_1": {
                "files": {
                    "core_data": "dataset_uuid/table/location_id=1/part_1.parquet"
                }
            }
        },
        "indices": {
            "product_id": {
                "1": ["part_1"],
                "2": ["part_1"],
                "100": ["part_1"],
                "34": ["part_1"],
            }
        },
    }
    store.put(
        "uuid.by-dataset-metadata.json", simplejson.dumps(meta_dct).encode("utf-8")
    )
    df = pd.DataFrame({"index": [1], "location_id": [1], "product_id": [1]})
    store_schema_metadata(make_meta(df, origin="core"), "uuid", store, "core_data")

    storage_key = "uuid/some_index.parquet"
    index2 = ExplicitSecondaryIndex(
        column="location_id",
        index_dct={1: ["part_1", "part_2"], 3: ["part_3"]},
        index_storage_key=storage_key,
        dtype=pa.int64(),
    )
    index2.store(store, "dataset_uuid")

    dmd = DatasetMetadata.load_from_store(store=store, uuid="uuid")
    assert "location_id" not in dmd.indices

    dmd = DatasetMetadata.load_from_store(
        store=store, uuid="uuid", load_all_indices=True
    )
    assert "location_id" in dmd.indices


def test_load_partition_indices_types(store):
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    table = "table"
    index_name = "location_id"
    index_value = 1
    meta_dct = {
        "dataset_metadata_version": 4,
        "dataset_uuid": dataset_uuid,
        "partitions": {
            "{index_name}={index_value}/part_1".format(
                index_name=index_name, index_value=index_value
            ): {
                "files": {
                    table: "{dataset_uuid}/{table}/location_id=1/part_1.parquet".format(
                        dataset_uuid=dataset_uuid, table=table
                    )
                }
            }
        },
    }
    store.put(
        "{dataset_uuid}.by-dataset-metadata.json".format(dataset_uuid=dataset_uuid),
        simplejson.dumps(meta_dct).encode(),
    )
    store_schema_metadata(
        make_meta(
            pd.DataFrame({index_name: pd.Series([index_value], dtype=int)}),
            origin="core",
        ),
        dataset_uuid,
        store,
        table,
    )
    dmd = DatasetMetadata.load_from_store(store=store, uuid=dataset_uuid)

    dmd = dmd.load_partition_indices()
    assert len(dmd.indices) == 1

    assert "location_id" in dmd.indices
    assert isinstance(dmd.indices["location_id"], PartitionIndex)

    idx = dmd.indices["location_id"]
    assert idx.dtype == pa.int64()
    assert idx.query(1) == ["location_id=1/part_1"]


def test_load_partition_keys(store):
    expected = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "uuid",
        "partitions": {
            "part_1": {
                "files": {"core_data": "uuid/table/index=1/index2=2/file.parquet"}
            },
            "part_2": {
                "files": {"core_data": "uuid/table/index=1/index2=2/file2.parquet"}
            },
        },
        "indices": {
            "product_id": {
                "1": ["part_1"],
                "2": ["part_2"],
                "100": ["part_1", "part_2"],
                "34": ["part_1"],
            },
            "location_id": {
                "1": ["part_1"],
                "2": ["part_2"],
                "3": ["part_1"],
                "4": ["part_2"],
            },
        },
    }
    store.put(
        "uuid.by-dataset-metadata.json", simplejson.dumps(expected).encode("utf-8")
    )
    df = pd.DataFrame(
        {"index": [1], "index2": [1], "product_id": [1], "location_id": [1]}
    )
    store_schema_metadata(make_meta(df, origin="core"), "uuid", store, "core_data")
    dmd = DatasetMetadata.load_from_store("uuid", store)
    assert dmd.partition_keys == ["index", "index2"]


def test_query_indices_external(store, metadata_version):
    expected = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": "uuid+namespace-attribute12_underscored",
        "partitions": {
            "part_1": {"files": {"core_data": "file.parquest"}},
            "part_2": {"files": {"core_data": "file2.parquest"}},
        },
        "indices": {
            "product_id": "uuid+namespace-attribute12_underscored.product_id.by-dataset-index.parquet",
            "location_id": {
                "1": ["part_1"],
                "2": ["part_2"],
                "3": ["part_1"],
                "4": ["part_2"],
            },
        },
    }
    store.put(
        "uuid+namespace-attribute12_underscored.by-dataset-metadata.json",
        simplejson.dumps(expected).encode("utf-8"),
    )
    df = pd.DataFrame(
        {
            "product_id": [1, 2, 100, 34],
            "partition": [
                np.array(["part_1"], dtype=object),
                np.array(["part_2"], dtype=object),
                np.array(["part_1", "part_2"], dtype=object),
                np.array(["part_1"], dtype=object),
            ],
        }
    )
    schema = pa.schema(
        [
            pa.field("partition", pa.list_(pa.string())),
            pa.field("product_id", pa.int64()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    store.put(
        "uuid+namespace-attribute12_underscored.product_id.by-dataset-index.parquet",
        buf.getvalue().to_pybytes(),
    )
    store_schema_metadata(
        make_meta(df, origin="core"),
        "uuid+namespace-attribute12_underscored",
        store,
        "core_data",
    )

    dmd = DatasetMetadata.load_from_store(
        "uuid+namespace-attribute12_underscored", store
    )

    dmd = dmd.load_index("product_id", store)
    assert dmd.query(product_id=2) == ["part_2"]
    dmd = dmd.load_all_indices(store)
    assert dmd.query(product_id=2, location_id=2) == ["part_2"]
    assert dmd.query(product_id=100, location_id=3) == ["part_1"]
    assert dmd.query(product_id=2, location_id=2, something_else="bla") == ["part_2"]

    additional_index = ExplicitSecondaryIndex.from_v2(
        "another_column", {"1": ["part_2", "part_3"]}
    )
    assert dmd.query(
        indices=[additional_index], another_column="1", product_id=2, location_id=2
    ) == ["part_2"]


def test_copy(frozen_time):
    ds = DatasetMetadata(
        uuid="uuid",
        partitions={"partition_label": {"files": {}}},
        metadata={"some": "metadata"},
        indices={
            "column": ExplicitSecondaryIndex(
                column="column", index_dct={1: ["partition_label"]}
            )
        },
        explicit_partitions=True,
        partition_keys=["P", "L"],
    )
    new_ds = ds.copy()
    # Check if the copy is identical
    assert new_ds == ds
    # ... but not the same object
    assert id(new_ds) != id(ds)

    new_ds = ds.copy(metadata={"new": "metadata"})
    assert id(new_ds) != id(ds)
    assert new_ds.metadata == {
        "new": "metadata",
        # The DatasetMetadata constructor ensure that the creation time is
        # always present.
        "creation_time": "2000-01-01T01:01:01.000001",
    }


def test_load_partition_indices_no_files(store):
    meta_dct = {
        "dataset_metadata_version": 4,
        "dataset_uuid": "dataset_uuid",
        "partitions": {"p1": {"files": {}}},
    }
    dmd = DatasetMetadata.load_from_dict(meta_dct, store)
    dmd = dmd.load_partition_indices()
    assert len(dmd.indices) == 0


def test_load_missing(store):
    with pytest.raises(KeyError) as exc:
        DatasetMetadata.load_from_store("not_there", store)
    assert exc.value.args == (
        "Dataset does not exist. Tried not_there.by-dataset-metadata.json and not_there.by-dataset-metadata.msgpack.zstd",
    )
