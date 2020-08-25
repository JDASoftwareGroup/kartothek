import os
import random
import tempfile
from urllib.parse import quote

import numpy as np
import pandas as pd
import simplejson
import storefact

from kartothek.core.common_metadata import (
    _get_common_metadata_key,
    make_meta,
    store_schema_metadata,
)
from kartothek.core.dataset import DatasetMetadata, create_partition_key, naming
from kartothek.core.urlencode import quote_indices, unquote_indices


def test_create_partition_key():
    key = create_partition_key(
        "my-uuid", "testtable", [("index1", "value1"), ("index2", "value2")]
    )
    assert key == "my-uuid/testtable/index1=value1/index2=value2/data"


def test_index_quote_roundtrip():
    indices = [
        (1, b"Muenchen"),
        ("location", b"Muenchen"),
        ("location", "München"),
        ("product", "å\\ øß"),
    ]
    expected = [
        ("1", "Muenchen"),
        ("location", "Muenchen"),
        ("location", "München"),
        ("product", "å\\ øß"),
    ]
    assert expected == unquote_indices(quote_indices(indices))


def testunquote_indices():
    index_strings = [
        "{}={}".format(quote("location"), quote("München".encode("utf-8"))),
        "{}={}".format(quote("product"), quote("å\\ øß".encode("utf-8"))),
    ]
    indices = unquote_indices(index_strings)
    assert indices == [("location", "München"), ("product", "å\\ øß")]


def test_dynamic_partitions(store):
    """
    Do not specify partitions in metadata, but read them dynamically from store
    """
    partition_suffix = "suffix"
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0")],
        "{}.parquet".format(partition_suffix),
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1")],
        "{}.parquet".format(partition_suffix),
    )
    partition0_ext = create_partition_key(
        dataset_uuid,
        "extension",
        [("location", "L-0")],
        "{}.parquet".format(partition_suffix),
    )
    partition1_ext = create_partition_key(
        dataset_uuid,
        "extension",
        [("location", "L-1")],
        "{}.parquet".format(partition_suffix),
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        "location=L-0/{}".format(partition_suffix): {
            "files": {"core": partition0_core, "extension": partition0_ext}
        },
        "location=L-1/{}".format(partition_suffix): {
            "files": {"core": partition1_core, "extension": partition1_ext}
        },
    }
    expected_indices = {
        "location": {
            "L-0": ["location=L-0/{}".format(partition_suffix)],
            "L-1": ["location=L-1/{}".format(partition_suffix)],
        }
    }

    # put two partitions for two tables each to store
    store.put(
        "{}{}.json".format(dataset_uuid, naming.METADATA_BASE_SUFFIX),
        simplejson.dumps(metadata).encode("utf-8"),
    )
    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store.put(partition0_ext, b"test")
    store.put(partition1_ext, b"test")
    store_schema_metadata(
        make_meta(
            pd.DataFrame({"location": ["L-0/{}".format(partition_suffix)]}),
            origin="stored",
        ),
        dataset_uuid,
        store,
        "core",
    )

    # instantiate metadata to write table metadatad
    core_schema = make_meta(
        pd.DataFrame(
            {
                "column_0": pd.Series([1], dtype=int),
                "column_1": pd.Series([1], dtype=int),
                "location": pd.Series(["str"]),
            }
        ),
        origin="core",
    )
    extension_schema = make_meta(
        pd.DataFrame(
            {
                "column_77": pd.Series([1], dtype=int),
                "column_78": pd.Series([1], dtype=int),
                "location": pd.Series(["str"]),
            }
        ),
        origin="extension",
    )
    store_schema_metadata(core_schema, dataset_uuid, store, "core")
    store_schema_metadata(extension_schema, dataset_uuid, store, "extension")
    dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
    # reload metadata to use table metadata
    dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
    dmd = dmd.load_partition_indices()

    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    assert dmd_dict["indices"] == expected_indices


def test_dynamic_partitions_multiple_indices(store):
    """
    Do not specify partitions in metadata, but read them dynamically from store
    """
    suffix = "suffix"
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0"), ("product", "P-0")],
        "{}.parquet".format(suffix),
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1"), ("product", "P-0")],
        "{}.parquet".format(suffix),
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        "location=L-0/product=P-0/{}".format(suffix): {
            "files": {"core": partition0_core}
        },
        "location=L-1/product=P-0/{}".format(suffix): {
            "files": {"core": partition1_core}
        },
    }
    expected_indices = {
        "location": {
            "L-0": ["location=L-0/product=P-0/{}".format(suffix)],
            "L-1": ["location=L-1/product=P-0/{}".format(suffix)],
        },
        "product": {
            "P-0": [
                "location=L-0/product=P-0/{}".format(suffix),
                "location=L-1/product=P-0/{}".format(suffix),
            ]
        },
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"], "product": ["P-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()
    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    # Sorting may differ in the index list. This is ok for runtime
    # but does produce flaky tests thus sort them.
    sorted_result = {
        column: {label: sorted(x) for label, x in index.items()}
        for column, index in dmd_dict["indices"].items()
    }
    assert sorted_result == expected_indices


def test_dynamic_partitions_with_garbage(store):
    """
    In case there are unknown files, dataset and indices still load correctly
    """
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition_suffix = "suffix"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0"), ("product", "P-0")],
        "{}.parquet".format(partition_suffix),
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1"), ("product", "P-0")],
        "{}.parquet".format(partition_suffix),
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        "location=L-0/product=P-0/{}".format(partition_suffix): {
            "files": {"core": partition0_core}
        },
        "location=L-1/product=P-0/{}".format(partition_suffix): {
            "files": {"core": partition1_core}
        },
    }
    expected_indices = {
        "location": {
            "L-0": ["location=L-0/product=P-0/{}".format(partition_suffix)],
            "L-1": ["location=L-1/product=P-0/{}".format(partition_suffix)],
        },
        "product": {
            "P-0": [
                "location=L-0/product=P-0/{}".format(partition_suffix),
                "location=L-1/product=P-0/{}".format(partition_suffix),
            ]
        },
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"], "product": ["P-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    # the following files are garbage and should not interfere with the indices and/or partitions
    for suffix in ["", ".json", ".msgpack", ".my_own_file_format"]:
        store.put("this_should_not_exist{}".format(suffix), b"ignore me")
        store.put(
            "{}/this_should_not_exist{}".format(dataset_uuid, suffix), b"ignore me"
        )
        store.put(
            "{}/{}/this_should_not_exist{}".format(dataset_uuid, "core", suffix),
            b"ignore me",
        )
        store.put(
            "{}/{}/location=L-0/this_should_not_exist{}".format(
                dataset_uuid, "core", suffix
            ),
            b"ignore me",
        )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()
    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    # Sorting may differ in the index list. This is ok for runtime
    # but does produce flaky tests thus sort them.
    sorted_result = {
        column: {label: sorted(x) for label, x in index.items()}
        for column, index in dmd_dict["indices"].items()
    }
    assert sorted_result == expected_indices


def test_dynamic_partitions_quote(store, metadata_version):
    """
    Do not specify partitions in metadata, but read them dynamically from store
    """
    dataset_uuid = "uuid-namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid, "core", [("location", "München")], "data.parquet"
    )
    partition1_core = create_partition_key(
        dataset_uuid, "core", [("location", "å\\ øß")], "data.parquet"
    )
    metadata = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": dataset_uuid,
    }
    expected_partitions = {
        "location=M%C3%BCnchen/data": {"files": {"core": partition0_core}},
        "location=%C3%A5%5C%20%C3%B8%C3%9F/data": {"files": {"core": partition1_core}},
    }
    expected_indices = {
        "location": {
            "München": ["location=M%C3%BCnchen/data"],
            "å\\ øß": ["location=%C3%A5%5C%20%C3%B8%C3%9F/data"],
        }
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()

    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    assert dmd_dict["indices"] == expected_indices


def test_dask_partitions(metadata_version):
    """
    Create partitions for one table with dask
    and check that it can be read with kartothek
    """
    import dask.dataframe

    bucket_dir = tempfile.mkdtemp()
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    os.mkdir("{}/{}".format(bucket_dir, dataset_uuid))
    table_dir = "{}/{}/core".format(bucket_dir, dataset_uuid)
    os.mkdir(table_dir)
    store = storefact.get_store_from_url("hfs://{}".format(bucket_dir))

    locations = ["L-{}".format(i) for i in range(2)]
    df = pd.DataFrame()
    for location in locations:
        core = pd.DataFrame(
            data={
                "date": np.array(
                    ["2017-11-23", "2017-11-23", "2017-11-24", "2017-11-24"]
                ),
                "product": np.array(["P-0", "P-1", "P-0", "P-1"]),
                "location": location,
                "value": np.array(random.sample(range(1, 100), 4)),
            }
        )
        df = pd.concat([df, core])

    ddf = dask.dataframe.from_pandas(df, npartitions=1)
    dask.dataframe.to_parquet(ddf, table_dir, partition_on=["location"])

    partition0 = "{}/core/location=L-0/part.0.parquet".format(dataset_uuid)
    partition1 = "{}/core/location=L-1/part.0.parquet".format(dataset_uuid)
    metadata = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": dataset_uuid,
    }
    expected_partitions = {
        "partitions": {
            "location=L-0": {"files": {"core": partition0}},
            "location=L-1": {"files": {"core": partition1}},
        }
    }
    expected_tables = {"tables": {"core": ["date", "product", "value"]}}

    store.put(
        "{}.by-dataset-metadata.json".format(dataset_uuid),
        simplejson.dumps(metadata).encode(),
    )

    metadata.update(expected_partitions)
    metadata.update(expected_tables)
    dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
    actual_partitions = dmd.to_dict()["partitions"]
    # we partition on location ID which has two values
    assert len(actual_partitions) == 2
    assert dmd.partition_keys == ["location"]


def test_overlap_keyspace(store, metadata_version):
    dataset_uuid1 = "uuid+namespace-attribute12_underscored"
    dataset_uuid2 = "uuid+namespace-attribute12_underscored_ext"
    table = "core"

    for dataset_uuid in (dataset_uuid1, dataset_uuid2):
        partition0 = "location=L-0"
        partition0_key = "{}/{}/{}/data.parquet".format(dataset_uuid, table, partition0)
        metadata = {
            "dataset_metadata_version": metadata_version,
            "dataset_uuid": dataset_uuid,
        }

        # put two partitions for two tables each to store
        store.put(
            "{}{}.json".format(dataset_uuid, naming.METADATA_BASE_SUFFIX),
            simplejson.dumps(metadata).encode("utf-8"),
        )
        store.put(partition0_key, b"test")
        store_schema_metadata(
            make_meta(pd.DataFrame({"location": ["L-0"]}), origin="1"),
            dataset_uuid,
            store,
            "core",
        )

    for dataset_uuid in (dataset_uuid1, dataset_uuid2):
        partition0_label = "location=L-0/data"
        partition0_key = "{}/{}/{}.parquet".format(
            dataset_uuid, table, partition0_label
        )
        expected_partitions = {"location=L-0/data": {"files": {"core": partition0_key}}}
        expected_indices = {"location": {"L-0": ["location=L-0/data"]}}
        assert DatasetMetadata.storage_keys(dataset_uuid, store) == [
            "{}{}.json".format(dataset_uuid, naming.METADATA_BASE_SUFFIX),
            _get_common_metadata_key(dataset_uuid, "core"),
            partition0_key,
        ]
        dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
        dmd = dmd.load_partition_indices()
        dmd_dict = dmd.to_dict()
        assert dmd_dict["partitions"] == expected_partitions
        assert dmd_dict["indices"] == expected_indices
