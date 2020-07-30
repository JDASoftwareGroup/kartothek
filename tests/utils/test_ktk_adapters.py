import dask.bag as db
import pandas as pd
import pandas.testing as pdt
import pytest

from kartothek.core.cube.constants import (
    KTK_CUBE_DF_SERIALIZER,
    KTK_CUBE_METADATA_STORAGE_FORMAT,
    KTK_CUBE_METADATA_VERSION,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.dask.bag import store_bag_as_dataset
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition
from kartothek.io_components.read import dispatch_metapartitions
from kartothek.utils.ktk_adapters import (
    get_dataset_columns,
    get_dataset_keys,
    get_dataset_schema,
    get_partition_dataframe,
    get_physical_partition_stats,
    metadata_factory_from_dataset,
)


@pytest.fixture(params=[True, False])
def cube_has_ts_col(request):
    return request.param


@pytest.fixture
def cube():
    return Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube",)


@pytest.fixture
def ds(function_store, cube_has_ts_col):
    dfs = [pd.DataFrame({"x": x, "p": [0, 1], "i": 0, "_foo": 0}) for x in [0, 1]]
    if cube_has_ts_col:
        for df in dfs:
            df["KLEE_TS"] = pd.Timestamp("2019-01-01")

    mps = [
        MetaPartition(
            label="mp{}".format(i),
            data={SINGLE_TABLE: df},
            metadata_version=KTK_CUBE_METADATA_VERSION,
        )
        .partition_on(["p"] + (["KLEE_TS"] if cube_has_ts_col else []))
        .build_indices(["i"])
        for i, df in enumerate(dfs)
    ]

    return store_bag_as_dataset(
        bag=db.from_sequence(mps, partition_size=1),
        store=function_store,
        dataset_uuid="uuid",
        partition_on=(["p"] + (["KLEE_TS"] if cube_has_ts_col else [])),
        metadata_storage_format=KTK_CUBE_METADATA_STORAGE_FORMAT,
        metadata_version=KTK_CUBE_METADATA_VERSION,
        df_serializer=KTK_CUBE_DF_SERIALIZER,
    ).compute()


def test_get_dataset_schema(ds):
    assert get_dataset_schema(ds) == ds.table_meta[SINGLE_TABLE]


def test_get_dataset_columns(ds):
    cols = get_dataset_columns(ds)
    assert cols == {"_foo", "i", "p", "x"}
    assert all(isinstance(col, str) for col in cols)


@pytest.mark.parametrize("load_schema", [False, True])
def test_metadata_factory_from_dataset_no_store(function_store, ds, load_schema):
    ds2 = DatasetMetadata.load_from_store(
        "uuid", function_store(), load_schema=load_schema
    )
    factory = metadata_factory_from_dataset(ds2, with_schema=load_schema)
    assert factory.dataset_metadata is ds2

    store = factory.store
    with pytest.raises(NotImplementedError):
        store.get("foo")


@pytest.mark.parametrize("load_schema", [False, True])
def test_metadata_factory_from_dataset_with_store(function_store, ds, load_schema):
    ds2 = DatasetMetadata.load_from_store(
        "uuid", function_store(), load_schema=load_schema
    )
    factory = metadata_factory_from_dataset(
        ds2, with_schema=load_schema, store=function_store
    )
    assert factory.dataset_metadata is ds2

    store = factory.store
    store.put("foo", b"bar")
    assert store.get("foo") == b"bar"


class TestGetDatasetKeys:
    def test_simple(self, function_store, ds):
        assert get_dataset_keys(ds) == set(function_store().keys())

    def test_ignores_untracked(self, function_store, ds):
        keys = set(function_store().keys())

        # irrelevant content
        function_store().put(ds.uuid + ".foo", b"")

        assert get_dataset_keys(ds) == keys

    def test_partition_indices_loaded(self, function_store, ds):
        ds = ds.load_partition_indices()

        assert get_dataset_keys(ds) == set(function_store().keys())

    def test_all_indices_loaded(self, function_store, ds):
        ds = ds.load_all_indices(function_store())

        assert get_dataset_keys(ds) == set(function_store().keys())


def test_get_physical_partition_stats(function_store, ds):
    mps_list = list(
        dispatch_metapartitions(
            dataset_uuid=ds.uuid, store=function_store, dispatch_by=["p"]
        )
    )
    assert len(mps_list) == 2

    for i, mps in enumerate(mps_list):
        actual = get_physical_partition_stats(mps, function_store)
        blobsize = sum(
            len(function_store().get(f))
            for f in function_store().iter_keys()
            if "p={}".format(i) in f
        )
        expected = {"partitions": 1, "files": 2, "rows": 2, "blobsize": blobsize}
        assert actual == expected


@pytest.mark.parametrize("different_partioning", [False, True])
def test_get_partition_dataframe(ds, cube, different_partioning, cube_has_ts_col):
    if different_partioning:
        cube = cube.copy(partition_columns=["missing"])

    df_expected = pd.DataFrame(
        data={
            "p": [0, 0, 1, 1],
            "partition": [
                "p=0/KLEE_TS=2019-01-01%2000%3A00%3A00/mp0",
                "p=0/KLEE_TS=2019-01-01%2000%3A00%3A00/mp1",
                "p=1/KLEE_TS=2019-01-01%2000%3A00%3A00/mp0",
                "p=1/KLEE_TS=2019-01-01%2000%3A00%3A00/mp1",
            ]
            if cube_has_ts_col
            else ["p=0/mp0", "p=0/mp1", "p=1/mp0", "p=1/mp1"],
        },
        columns=["p", "partition"],
    ).set_index("partition")

    ds = ds.load_partition_indices()

    df_actual = get_partition_dataframe(ds, cube)
    pdt.assert_frame_equal(df_actual, df_expected)
