from kartothek.core.dataset import DatasetMetadata
from kartothek.utils.ktk_adapters import get_dataset_columns


def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "index")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store(cli, skv):
    result = cli("--store=cubi", "my_cube", "index")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "index")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_fail_ds_not_found(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "index", "foo", "v1")
    assert result.exit_code == 2
    assert (
        "Error: Could not find dataset foo, known datasets are enrich, source"
        in result.output
    )


def test_fail_column_not_found(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "index", "source", "foo")
    assert result.exit_code == 2

    assert "Error: Could not find column foo" in result.output


def test_simple(cli, built_cube, skv, store):
    ds = DatasetMetadata.load_from_store(built_cube.ktk_dataset_uuid("source"), store)
    assert "v1" not in ds.indices

    result = cli("--store=cubes", "my_cube", "index", "source", "v1")
    assert result.exit_code == 0

    ds = DatasetMetadata.load_from_store(built_cube.ktk_dataset_uuid("source"), store)
    assert "v1" in ds.indices


def test_all(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "index", "source", "*")
    assert result.exit_code == 0

    ds = DatasetMetadata.load_from_store(built_cube.ktk_dataset_uuid("source"), store)
    assert set(ds.indices.keys()) == set(get_dataset_columns(ds))
