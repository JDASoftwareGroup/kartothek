import pandas as pd
import pytest

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.io.eager_cube import extend_cube
from kartothek.utils.ktk_adapters import get_dataset_keys


def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "copy", "--src_store=cubes2")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store_src(cli, skv):
    result = cli("--store=cubi", "my_cube", "copy", "--tgt_store=cubes2")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_wrong_store_tgt(cli, skv, built_cube):
    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubi2")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi2 in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "copy")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_fail_same_store_no_overwrite(cli, built_cube, skv, store):
    keys = set(store.keys())

    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubes")
    assert result.exit_code == 2
    assert "Error: Source and target store must be different." in result.output

    assert set(store.keys()) == keys


def test_fail_same_store_overwrite(cli, built_cube, skv, store):
    keys = set(store.keys())

    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubes", "--overwrite")
    assert result.exit_code == 2
    assert "Error: Source and target store must be different." in result.output

    assert set(store.keys()) == keys


def test_simple(cli, built_cube, skv, store, store2):
    assert set(store2.keys()) == set()

    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubes2")
    assert result.exit_code == 0

    assert set(store2.keys()) == set(store.keys())
    for k in sorted(store.keys()):
        assert store.get(k) == store2.get(k)


def test_fail_overwrite(cli, built_cube, built_cube2, skv):
    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubes2")
    assert result.exit_code == 2
    assert (
        'Error: Failed to copy cube: Dataset "my_cube++source" exists in target store but overwrite was set to False'
        in result.output
    )


def test_overwrite_cleanup(cli, built_cube, built_cube2, skv, store, store2):
    assert len(set(store2.keys())) > 0

    result = cli(
        "--store=cubes", "my_cube", "copy", "--tgt_store=cubes2", "--overwrite"
    )
    assert result.exit_code == 0

    assert set(store2.keys()) == set(store.keys())

    for k in sorted(store.keys()):
        assert store.get(k) == store2.get(k)


def test_overwrite_nocleanup(cli, built_cube, built_cube2, skv, store, store2):
    assert len(set(store2.keys())) > 0

    result = cli(
        "--store=cubes",
        "my_cube",
        "copy",
        "--tgt_store=cubes2",
        "--overwrite",
        "--no-cleanup",
    )
    assert result.exit_code == 0

    keys1 = set(store.keys())
    keys2 = set(store2.keys())

    assert keys1 - keys2 == set(), "not all keys got copied"
    assert len(keys2 - keys1) > 0, "target cube seemed to be deleted before copy"

    for k in sorted(store.keys()):
        assert store.get(k) == store2.get(k)


@pytest.mark.parametrize(
    "key_pattern,copy_tables",
    [
        ("enrich,source", ["enrich", "source"]),
        ("en*,source", ["enrich", "source"]),
        ("enri?h,source", ["enrich", "source"]),
        ("en[a-z]ich,source", ["enrich", "source"]),
        ("my[a-z]*,source", ["mytable", "source"]),
        ("so*", ["source"]),
    ],
)
def test_partial_copy_include_pattern(
    cli, built_cube, skv, store, store2, key_pattern, copy_tables
):
    extend_cube(
        data={
            "mytable": pd.DataFrame(
                {
                    "x": [0, 1],
                    "y": [0, 0],
                    "p": 0,
                    "q": ["a", "a"],
                    "mycolumn": ["a", "b"],
                }
            )
        },
        cube=built_cube,
        store=store,
    )
    copied_datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=copy_tables,
    )
    copy_keys = set()
    for name in copy_tables:
        copy_keys |= get_dataset_keys(copied_datasets[name])

    result = cli(
        "--store=cubes",
        "my_cube",
        "copy",
        "--tgt_store=cubes2",
        "--include=" + key_pattern,
    )
    assert result.exit_code == 0
    assert set(store2.keys()) == copy_keys


def test_partial_copy_include_pattern_nomatch(cli, built_cube, skv, store, store2):
    copied_datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=["source"],
    )
    copy_keys = get_dataset_keys(copied_datasets["source"])  # noqa
    result = cli(
        "--store=cubes", "my_cube", "copy", "--tgt_store=cubes2", "--include=x*,source"
    )
    assert result.exit_code == 2

    assert "Error: Could not find dataset x*" in result.output


@pytest.mark.parametrize(
    "exclude_pattern,copy_tables",
    [
        ("enrich", ["mytable", "source"]),
        ("en*", ["mytable", "source"]),
        ("enri?h", ["mytable", "source"]),
        ("en[a-z]ich", ["mytable", "source"]),
        ("my[a-z]*", ["enrich", "source"]),
        ("my*,e*", ["source"]),
    ],
)
def test_partial_copy_exclude_pattern(
    cli, built_cube, skv, store, store2, exclude_pattern, copy_tables
):
    extend_cube(
        data={
            "mytable": pd.DataFrame(
                {
                    "x": [0, 1],
                    "y": [0, 0],
                    "p": 0,
                    "q": ["a", "a"],
                    "mycolumn": ["a", "b"],
                }
            )
        },
        cube=built_cube,
        store=store,
    )
    copied_datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=copy_tables,
    )
    copy_keys = set()
    for name in copy_tables:
        copy_keys |= get_dataset_keys(copied_datasets[name])
    result = cli(
        "--store=cubes",
        "my_cube",
        "copy",
        "--tgt_store=cubes2",
        "--exclude=" + exclude_pattern,
    )
    assert result.exit_code == 0
    assert set(store2.keys()) == copy_keys


def test_partial_copy_exclude_pattern_nomatch(cli, built_cube, skv, store, store2):
    result = cli(
        "--store=cubes", "my_cube", "copy", "--tgt_store=cubes2", "--exclude=x*"
    )
    assert result.exit_code == 2

    assert "Error: Could not find dataset x*" in result.output


@pytest.mark.parametrize(
    "exclude_pattern,include_pattern,copy_tables",
    [
        ("enrich", "source", ["source"]),
        ("en*", "so*", ["source"]),
        ("enri?h", "sour?e", ["source"]),
        ("en[a-z]ich", "so[a-z]rce", ["source"]),
        ("en[a-z]*", "sour[a-z]*", ["source"]),
        ("enrich,mytable", "mytable,source", ["source"]),
        ("en[a-z]*,mytable", "mytable,source", ["source"]),
        ("en*", "mytable,sour*", ["mytable", "source"]),
    ],
)
def test_partial_copy_include_exclude_pattern(
    cli, built_cube, skv, store, store2, include_pattern, exclude_pattern, copy_tables
):
    extend_cube(
        data={
            "mytable": pd.DataFrame(
                {
                    "x": [0, 1],
                    "y": [0, 0],
                    "p": 0,
                    "q": ["a", "a"],
                    "mycolumn": ["a", "b"],
                }
            )
        },
        cube=built_cube,
        store=store,
    )
    copied_datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=copy_tables,
    )
    copy_keys = set()
    for name in copy_tables:
        copy_keys |= get_dataset_keys(copied_datasets[name])
    result = cli(
        "--store=cubes",
        "my_cube",
        "copy",
        "--tgt_store=cubes2",
        "--include=" + include_pattern,
        "--exclude=" + exclude_pattern,
    )
    assert result.exit_code == 0
    assert set(store2.keys()) == copy_keys


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_azure(cli, built_cube_azure, skv_azure, azurestore, azurestore2, caplog):
    assert set(azurestore2.keys()) == set()

    result = cli("--store=cubes", "my_cube", "copy", "--tgt_store=cubes2")
    assert result.exit_code == 0

    assert set(azurestore2.keys()) == set(azurestore.keys())
    assert len(caplog.records) == 0
