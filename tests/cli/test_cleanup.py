def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "cleanup")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store(cli, skv):
    result = cli("--store=cubi", "my_cube", "cleanup")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "cleanup")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_simple(cli, built_cube, skv, store):
    key = built_cube.ktk_dataset_uuid(built_cube.seed_dataset) + "/foo"
    store.put(key, b"")

    result = cli("--store=cubes", "my_cube", "cleanup")
    assert result.exit_code == 0

    assert key not in set(store.keys())
