from collections import Counter

import pandas as pd
import pytest
from simplekv import KeyValueStore

from kartothek.api.discover import (
    discover_cube,
    discover_datasets,
    discover_datasets_unchecked,
    discover_ktk_cube_dataset_ids,
)
from kartothek.core.cube.constants import (
    KTK_CUBE_DF_SERIALIZER,
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
    KTK_CUBE_METADATA_STORAGE_FORMAT,
    KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    KTK_CUBE_METADATA_VERSION,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.uuid import gen_uuid
from kartothek.io.eager import (
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
)
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition


@pytest.fixture
def cube():
    return Cube(
        dimension_columns=["x", "y"],
        partition_columns=["p", "q"],
        uuid_prefix="cube",
        index_columns=["i1"],
        seed_dataset="myseed",
    )


def store_data(
    cube,
    function_store,
    df,
    name,
    partition_on="default",
    metadata_version=KTK_CUBE_METADATA_VERSION,
    metadata_storage_format=KTK_CUBE_METADATA_STORAGE_FORMAT,
    metadata=None,
    overwrite=False,
    new_ktk_cube_metadata=True,
    write_suppress_index_on=True,
):
    if partition_on == "default":
        partition_on = cube.partition_columns

    if isinstance(df, pd.DataFrame):
        mp = MetaPartition(
            label=gen_uuid(), data={SINGLE_TABLE: df}, metadata_version=metadata_version
        )

        indices_to_build = set(cube.index_columns) & set(df.columns)
        if name == cube.seed_dataset:
            indices_to_build |= set(cube.dimension_columns) - set(
                cube.suppress_index_on
            )
        mp = mp.build_indices(indices_to_build)
        dfs = mp
    else:
        assert isinstance(df, MetaPartition)
        assert df.metadata_version == metadata_version
        dfs = df

    if metadata is None:
        metadata = {
            KTK_CUBE_METADATA_DIMENSION_COLUMNS: cube.dimension_columns,
            KTK_CUBE_METADATA_KEY_IS_SEED: (name == cube.seed_dataset),
        }
        if new_ktk_cube_metadata:
            metadata.update(
                {KTK_CUBE_METADATA_PARTITION_COLUMNS: cube.partition_columns}
            )
        if write_suppress_index_on:
            metadata.update(
                {KTK_CUBE_METADATA_SUPPRESS_INDEX_ON: list(cube.suppress_index_on)}
            )

    return store_dataframes_as_dataset(
        store=function_store,
        dataset_uuid=cube.ktk_dataset_uuid(name),
        dfs=dfs,
        partition_on=list(partition_on) if partition_on else None,
        metadata_storage_format=metadata_storage_format,
        metadata_version=metadata_version,
        df_serializer=KTK_CUBE_DF_SERIALIZER,
        metadata=metadata,
        overwrite=overwrite,
    )


def assert_datasets_equal(left, right):
    assert set(left.keys()) == set(right.keys())

    for k in left.keys():
        ds_l = left[k]
        ds_r = right[k]

        assert ds_l.uuid == ds_r.uuid


def assert_dataset_issubset(superset, subset):
    assert set(subset.keys()).issubset(set(superset.keys()))
    for k in subset.keys():
        assert subset[k].uuid == superset[k].uuid


def test_discover_ktk_cube_dataset_ids(function_store):
    cube = Cube(
        dimension_columns=["dim"],
        partition_columns=["part"],
        uuid_prefix="cube",
        seed_dataset="seed",
    )
    ktk_cube_dataset_ids = ["A", "B", "C"]
    for ktk_cube_id in ktk_cube_dataset_ids:
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"dim": [0], "part": [0]}),
            name=ktk_cube_id,
        )
    collected_ktk_cube_dataset_ids = discover_ktk_cube_dataset_ids(
        cube.uuid_prefix, function_store()
    )
    assert collected_ktk_cube_dataset_ids == set(ktk_cube_dataset_ids)


class TestDiscoverDatasetsUnchecked:
    def test_simple(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name="enrich",
            ),
        }

        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_no_seed(self, cube, function_store):
        expected = {
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name="enrich",
            )
        }
        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_other_files(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            )
        }

        function_store().put(cube.ktk_dataset_uuid("enrich") + "/foo", b"")

        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_no_common_metadata(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            )
        }

        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name="enrich",
        )
        keys = set(function_store().keys())
        metadata_key = cube.ktk_dataset_uuid("enrich") + ".by-dataset-metadata.json"
        assert metadata_key in keys
        for k in keys:
            if (k != metadata_key) and k.startswith(cube.ktk_dataset_uuid("enrich")):
                function_store().delete(k)

        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_filter_partial_datasets_found(self, cube, function_store):
        enrich_dataset = store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name="enrich",
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name="mytable",
        )
        expected = {"enrich": enrich_dataset}
        actual = discover_datasets_unchecked(
            cube.uuid_prefix, function_store, filter_ktk_cube_dataset_ids=["enrich"]
        )
        assert_dataset_issubset(actual, expected)

    def test_filter_no_datasets_found(self, cube, function_store):
        actual = discover_datasets_unchecked(
            cube.uuid_prefix, function_store, filter_ktk_cube_dataset_ids=["enrich"]
        )
        assert actual == {}

    def test_msgpack_clean(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name="enrich",
                metadata_storage_format="msgpack",
            ),
        }

        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_msgpack_priority(self, cube, function_store):
        """
        json metadata files have priority in kartothek, so the disovery should respect this
        """
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": [0]}),
            name=cube.seed_dataset,
            metadata_storage_format="msgpack",
        )
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v2": [0]}),
                name=cube.seed_dataset,
                overwrite=True,
            )
        }
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v3": [0]}),
            name=cube.seed_dataset,
            metadata_storage_format="msgpack",
            overwrite=True,
        )

        actual = discover_datasets_unchecked(cube.uuid_prefix, function_store)
        assert_datasets_equal(actual, expected)

    def test_msgpack_efficiency(self, cube, function_store):
        """
        We should only iterate over the store once, even though we are looking for 2 suffixes.

        Furthermore, we must only load every dataset once.
        """
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            metadata_storage_format="msgpack",
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            overwrite=True,
        )

        class StoreMock(KeyValueStore):
            def __init__(self, store):
                self._store = store
                self._iter_keys_called = 0
                self._iter_prefixes_called = 0
                self._get_called = Counter()

            def iter_keys(self, prefix=""):
                self._iter_keys_called += 1
                return self._store.iter_keys(prefix)

            def iter_prefixes(self, delimiter, prefix=""):
                self._iter_prefixes_called += 1
                return self._store.iter_prefixes(delimiter, prefix)

            def get(self, key):
                self._get_called[key] += 1
                return self._store.get(key)

        store = StoreMock(function_store())

        discover_datasets_unchecked(cube.uuid_prefix, store)

        assert store._iter_keys_called == 0
        assert store._iter_prefixes_called == 1
        assert max(store._get_called.values()) == 1


class TestDiscoverDatasets:
    def test_seed_only(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            )
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_2_datasets(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_partitions_superset(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
                partition_on=["p", "q", "v1"],
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_raises_no_seed(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert str(exc.value) == 'Seed data ("myseed") is missing.'

    def test_raises_wrong_partition_on_seed_other(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0]}),
            name=cube.seed_dataset,
            partition_on=["p"],
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value) == 'Seed dataset "myseed" has missing partition columns: q'
        )

    def test_partition_on_nonseed_no_part(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "v1": [0]}),
                name="enrich",
                partition_on=[],
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_raises_wrong_metadata_version(self, cube, function_store):
        with pytest.raises(
            NotImplementedError, match="Minimal supported metadata version is"
        ):
            store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
                metadata_version=2,
                partition_on=None,
            )

    def test_raises_wrong_table(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={"foo": pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]})},
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value)
            == "Invalid datasets because table is wrong. Expected table: myseed (foo)"
        )

    def test_raises_extra_table(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={
                    SINGLE_TABLE: pd.DataFrame(
                        {"x": [0], "y": [0], "p": [0], "q": [0]}
                    ),
                    "foo": pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                },
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ).build_indices(["x", "y"]),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value)
            == "Invalid datasets because table is wrong. Expected table: myseed (foo, table)"
        )

    def test_raises_dtypes(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0.0], "p": [0], "q": [0], "v1": 100}),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert 'Found incompatible entries for column "y"' in str(exc.value)

    def test_raises_overlap(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert "Found columns present in multiple datasets" in str(exc.value)

    def test_raises_partition_on_overlap(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "v1": 100}),
            name="enrich",
            partition_on=["v1"],
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert "Found columns present in multiple datasets" in str(exc.value)

    def test_raises_missing_dimension_columns(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={SINGLE_TABLE: pd.DataFrame({"x": [0], "p": [0], "q": [0]})},
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ).build_indices(["x"]),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value) == 'Seed dataset "myseed" has missing dimension columns: y'
        )

    def test_raises_no_dimension_columns(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"p": [0], "q": [0], "v2": 100}),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value)
            == 'Dataset "enrich" must have at least 1 of the following dimension columns: x, y'
        )

    def test_raises_dimension_index_missing(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={
                    SINGLE_TABLE: pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]})
                },
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value)
            == 'ExplicitSecondaryIndex "x" is missing in dataset "myseed".'
        )

    def test_raises_other_index_missing(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={
                    SINGLE_TABLE: pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]})
                },
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ).build_indices(["x", "y"]),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=MetaPartition(
                label=gen_uuid(),
                data={
                    SINGLE_TABLE: pd.DataFrame(
                        {"x": [0], "y": [0], "p": [0], "q": [0], "i1": [1337]}
                    )
                },
                metadata_version=KTK_CUBE_METADATA_VERSION,
            ),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value)
            == 'ExplicitSecondaryIndex or PartitionIndex "i1" is missing in dataset "enrich".'
        )

    def test_accepts_addional_indices(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {"x": [0], "y": [0], "p": [0], "q": [0], "v1": [0]}
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ).build_indices(["x", "y", "v1"]),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {
                                "x": [0],
                                "y": [0],
                                "p": [0],
                                "q": [0],
                                "i1": [1337],
                                "v2": [42],
                            }
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ).build_indices(["i1", "x", "v2"]),
                name="enrich",
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_accepts_partition_index_for_index(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {"x": [0], "y": [0], "i1": [1337], "v2": [42]}
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ),
                name="enrich",
                partition_on=["i1"],
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_raises_unspecified_partition_columns(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            partition_on=["p", "q"],
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": [0]}),
            name="enrich",
            partition_on=["q"],
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store)
        assert (
            str(exc.value) == "Unspecified but provided partition columns in enrich: p"
        )

    def test_accepts_projected_datasets(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {"x": [0], "y": [0], "p": [0], "q": [0]}
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ).build_indices(["x", "y"]),
                name=cube.seed_dataset,
            ),
            "x": store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {"x": [0], "p": [0], "q": [0], "v1": [42]}
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ),
                name="x",
            ),
            "y": store_data(
                cube=cube,
                function_store=function_store,
                df=MetaPartition(
                    label=gen_uuid(),
                    data={
                        SINGLE_TABLE: pd.DataFrame(
                            {"y": [0], "p": [0], "q": [0], "v2": [42]}
                        )
                    },
                    metadata_version=KTK_CUBE_METADATA_VERSION,
                ),
                name="y",
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_filter_basic(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
            ),
        }
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v2": 100}),
            name="foo",
        )
        actual = discover_datasets(cube, function_store, {"myseed", "enrich"})
        assert_datasets_equal(actual, expected)

    def test_filter_ignores_invalid(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
            ),
        }
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {
                    "x": [0],
                    "y": [0],
                    "p": [0],
                    "q": [0],
                    "v1": 100,  # overlapping payload
                }
            ),
            name="foo",
        )
        actual = discover_datasets(cube, function_store, {"myseed", "enrich"})
        assert_datasets_equal(actual, expected)

    def test_filter_missing(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store, {"myseed", "enrich"})
        assert (
            str(exc.value) == "Could not find the following requested datasets: enrich"
        )

    def test_filter_empty(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(cube, function_store, {})
        assert str(exc.value) == 'Seed data ("myseed") is missing.'

    def test_raises_partial_datasets_found(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name="enrich",
        )
        with pytest.raises(ValueError) as exc:
            discover_datasets(
                cube,
                function_store,
                filter_ktk_cube_dataset_ids=["enrich", "non_existing_table"],
            )
        assert (
            str(exc.value)
            == "Could not find the following requested datasets: non_existing_table"
        )

    def test_raises_no_datasets_found(self, cube, function_store):
        with pytest.raises(ValueError) as exc:
            discover_datasets(
                cube,
                function_store,
                filter_ktk_cube_dataset_ids=["enrich", "non_existing_table"],
            )
        assert (
            str(exc.value)
            == "Could not find the following requested datasets: enrich, non_existing_table"
        )

    def test_msgpack(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
                metadata_storage_format="msgpack",
            ),
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)

    def test_empty_dataset(self, cube, function_store):
        expected = {
            cube.seed_dataset: store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
                name=cube.seed_dataset,
            ),
            "enrich": store_data(
                cube=cube,
                function_store=function_store,
                df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "v1": 100}),
                name="enrich",
                metadata_storage_format="msgpack",
            ),
        }
        expected = {
            filter_ktk_cube_dataset_id: update_dataset_from_dataframes(
                [], store=function_store, dataset_uuid=ds.uuid, delete_scope=[{}]
            )
            for filter_ktk_cube_dataset_id, ds in expected.items()
        }
        actual = discover_datasets(cube, function_store)
        assert_datasets_equal(actual, expected)


class TestDiscoverCube:
    def test_seed_only(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "i1": [0]}),
            name=cube.seed_dataset,
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube
        assert set(datasets.keys()) == {cube.seed_dataset}
        ds = datasets[cube.seed_dataset]
        assert ds.primary_indices_loaded

    def test_without_partition_timestamp_metadata(self, cube, function_store):
        # test discovery of a cube without metadata keys
        # "KLEE_TS" and KTK_CUBE_METADATA_PARTITION_COLUMNS still works
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {
                    "x": [0],
                    "y": [0],
                    "p": [0],
                    "q": [0],
                    "KLEE_TS": [pd.Timestamp("2000")],
                    "i1": [0],
                }
            ),
            partition_on=["p", "q", "KLEE_TS"],
            name=cube.seed_dataset,
            new_ktk_cube_metadata=False,
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube
        assert set(datasets.keys()) == {cube.seed_dataset}

    def test_reads_suppress_index(self, cube, function_store):
        cube = cube.copy(suppress_index_on=cube.dimension_columns)
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "i1": [0]}),
            name=cube.seed_dataset,
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube

    def test_reads_suppress_index_default(self, cube, function_store):
        # test that reading also works for old metadata that does not contain the suppress_index_on method.
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "i1": [0]}),
            name=cube.seed_dataset,
            write_suppress_index_on=False,
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube

    def test_multiple(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "p": [0], "q": [0], "i1": [0]}),
            name="enrich",
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube
        assert set(datasets.keys()) == {cube.seed_dataset, "enrich"}
        ds_seed = datasets[cube.seed_dataset]
        assert ds_seed.primary_indices_loaded
        ds_enrich = datasets["enrich"]
        assert ds_enrich.primary_indices_loaded

    def test_partitions_superset(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "p": [0], "q": [0], "i1": [0], "v1": [0]}),
            name="enrich",
            partition_on=["p", "q", "v1"],
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube
        assert set(datasets.keys()) == {cube.seed_dataset, "enrich"}
        ds_seed = datasets[cube.seed_dataset]
        assert ds_seed.primary_indices_loaded
        ds_enrich = datasets["enrich"]
        assert ds_enrich.primary_indices_loaded

    def test_raises_no_seed(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0], "i1": [0]}),
            name=cube.seed_dataset,
            metadata={},
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert str(exc.value) == 'Could not find seed dataset for cube "cube".'

    def test_raises_multiple_seeds(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            metadata={KTK_CUBE_METADATA_KEY_IS_SEED: True},
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "p": [0], "q": [0], "i1": [0]}),
            name="enrich",
            metadata={KTK_CUBE_METADATA_KEY_IS_SEED: True},
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert (
            str(exc.value)
            == 'Found multiple possible seed datasets for cube "cube": enrich, myseed'
        )

    def test_raises_dimension_columns_missing(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            metadata={KTK_CUBE_METADATA_KEY_IS_SEED: True},
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert (
            str(exc.value)
            == 'Could not recover dimension columns from seed dataset ("myseed") of cube "cube".'
        )

    def test_raises_partition_keys_missing_old_metadata(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            partition_on=None,
            new_ktk_cube_metadata=False,
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert str(exc.value) == 'Seed dataset ("myseed") has no partition keys.'

    def test_raises_partition_keys_missing_seed_other(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            partition_on=None,
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert (
            str(exc.value)
            == 'Seed dataset "myseed" has missing partition columns: p, q'
        )

    def test_partition_keys_no_nonseed_other(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            partition_on=["p", "q"],
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "i1": [0], "v1": [0]}),
            name="enrich",
            partition_on=[],
        )
        cube_actual, datasets = discover_cube(cube.uuid_prefix, function_store)
        assert cube_actual == cube
        assert set(datasets.keys()) == {cube.seed_dataset, "enrich"}
        ds_seed = datasets[cube.seed_dataset]
        assert ds_seed.primary_indices_loaded
        ds_enrich = datasets["enrich"]
        assert (not ds_enrich.partition_keys) or ds_enrich.primary_indices_loaded

    def test_raises_partition_keys_impossible_old_metadata(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {
                    "x": [0],
                    "y": [0],
                    "p": [0],
                    "q": [0],
                    "KLEE_TS": [pd.Timestamp("2000")],
                }
            ),
            partition_on=["KLEE_TS"],
            name=cube.seed_dataset,
            new_ktk_cube_metadata=False,
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert (
            str(exc.value)
            == 'Seed dataset ("myseed") has only a single partition key (KLEE_TS) but should have at least 2.'
        )

    def test_timestamp_col_compat(self, cube, function_store):
        """
        Tests that cubes are still readable after timestamp removal.
        """
        metadata_dimension_columns_old = "klee_dimension_columns"
        metadata_is_seed_old = "klee_is_seed"
        metadata_partition_columns_old = "klee_partition_columns"
        metadata_timestamp_column_old = "klee_timestamp_column"
        timestamp_column_old = "KLEE_TS"

        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {
                    "x": [0],
                    "y": [0],
                    "p": [0],
                    "q": [0],
                    timestamp_column_old: [pd.Timestamp("2000")],
                    "i1": [0],
                    "a": [0],
                }
            ),
            partition_on=["p", "q", timestamp_column_old],
            name=cube.seed_dataset,
            metadata={
                metadata_dimension_columns_old: cube.dimension_columns,
                metadata_is_seed_old: True,
                metadata_partition_columns_old: cube.partition_columns,
                metadata_timestamp_column_old: timestamp_column_old,
            },
        )
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {
                    "x": [0],
                    "y": [0],
                    "p": [0],
                    "q": [0],
                    timestamp_column_old: [pd.Timestamp("2000")],
                    "b": [0],
                }
            ),
            partition_on=["p", "q", timestamp_column_old],
            name="enrich",
            metadata={
                metadata_dimension_columns_old: cube.dimension_columns,
                metadata_is_seed_old: False,
                metadata_partition_columns_old: cube.partition_columns,
                metadata_timestamp_column_old: timestamp_column_old,
            },
        )

        cube_discoverd, datasets_discovered = discover_cube(
            cube.uuid_prefix, function_store
        )
        assert cube == cube_discoverd
        assert set(datasets_discovered.keys()) == {cube.seed_dataset, "enrich"}

    def test_raises_timestamp_col_is_not_ktk_cube_ts(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame(
                {"x": [0], "y": [0], "p": [0], "q": [0], "ts": [pd.Timestamp("2000")]}
            ),
            partition_on=["p", "q", "ts"],
            name=cube.seed_dataset,
            new_ktk_cube_metadata=False,
        )
        with pytest.raises(
            NotImplementedError,
            match="Can only read old cubes if the timestamp column is 'KLEE_TS', but 'ts' was detected.",
        ):
            discover_cube(cube.uuid_prefix, function_store)

    def test_raises_partition_keys_impossible(self, cube, function_store):
        store_data(
            cube=cube,
            function_store=function_store,
            df=pd.DataFrame({"x": [0], "y": [0], "p": [0], "q": [0]}),
            name=cube.seed_dataset,
            partition_on=[],
        )
        with pytest.raises(ValueError) as exc:
            discover_cube(cube.uuid_prefix, function_store)
        assert (
            str(exc.value)
            == 'Seed dataset "myseed" has missing partition columns: p, q'
        )
