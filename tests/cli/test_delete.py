import pytest

from kartothek.api.discover import discover_datasets_unchecked
from kartothek.utils.ktk_adapters import get_dataset_keys


def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "delete")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store(cli, skv):
    result = cli("--store=cubi", "my_cube", "delete")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "delete")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_simple(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "delete")
    assert result.exit_code == 0

    assert set(store.keys()) == set()


@pytest.mark.parametrize(
    "key_pattern,delete_tables",
    [
        ("enrich", ["enrich"]),
        ("en*", ["enrich"]),
        ("enri?h", ["enrich"]),
        ("en[a-z]ich", ["enrich"]),
        ("en[a-z]*", ["enrich"]),
        ("enrich,source", ["enrich", "source"]),
        ("en[a-z]*,source", ["enrich", "source"]),
    ],
)
def test_partial_delete_include_pattern(
    cli, built_cube, skv, store, key_pattern, delete_tables
):
    datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=delete_tables,
    )
    delete_keys = set()
    for name in delete_tables:
        delete_keys |= get_dataset_keys(datasets[name])
    all_keys = set(store.keys())
    result = cli("--store=cubes", "my_cube", "delete", "--include=" + key_pattern)
    assert result.exit_code == 0
    assert set(store.keys()) == all_keys - delete_keys


def test_partial_delete_include_pattern_nomatch(cli, built_cube, skv, store):
    all_keys = set(store.keys())  # noqa
    result = cli("--store=cubes", "my_cube", "delete", "--include=x*")
    assert result.exit_code == 2

    assert "Error: Could not find dataset x*" in result.output


@pytest.mark.parametrize(
    "exclude_pattern,delete_tables",
    [
        ("enrich", ["source"]),
        ("en*", ["source"]),
        ("enri?h", ["source"]),
        ("en[a-z]ich", ["source"]),
        ("en[a-z]*", ["source"]),
        ("enrich,source", []),
        ("en[a-z]*,source", []),
    ],
)
def test_partial_delete_exclude_pattern(
    cli, built_cube, skv, store, exclude_pattern, delete_tables
):
    datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=delete_tables,
    )
    delete_keys = set()
    for name in delete_tables:
        delete_keys |= get_dataset_keys(datasets[name])
    all_keys = set(store.keys())
    result = cli("--store=cubes", "my_cube", "delete", "--exclude=" + exclude_pattern)
    assert result.exit_code == 0
    assert set(store.keys()) == all_keys - delete_keys


def test_partial_delete_exclude_pattern_nomatch(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "delete", "--exclude=x*")
    assert result.exit_code == 2

    assert "Error: Could not find dataset x*" in result.output


@pytest.mark.parametrize(
    "exclude_pattern,include_pattern,delete_tables",
    [
        ("enrich", "source", ["source"]),
        ("en*", "so*", ["source"]),
        ("enri?h", "sour?e", ["source"]),
        ("en[a-z]ich", "so[a-z]rce", ["source"]),
        ("en[a-z]*", "sour[a-z]*", ["source"]),
        ("enrich,source", "enrich,source", []),
        ("en[a-z]*,source", "en[a-z]*,source", []),
    ],
)
def test_partial_delete_include_exclude_pattern(
    cli, built_cube, skv, store, include_pattern, exclude_pattern, delete_tables
):
    datasets = discover_datasets_unchecked(
        uuid_prefix=built_cube.uuid_prefix,
        store=store,
        filter_ktk_cube_dataset_ids=delete_tables,
    )
    delete_keys = set()
    for name in delete_tables:
        delete_keys |= get_dataset_keys(datasets[name])
    all_keys = set(store.keys())  # noqa
    result = cli(
        "--store=cubes",
        "my_cube",
        "delete",
        "--include=" + include_pattern,
        "--exclude=" + exclude_pattern,
    )
    assert result.exit_code == 0
    assert set(store.keys()) == all_keys - delete_keys
