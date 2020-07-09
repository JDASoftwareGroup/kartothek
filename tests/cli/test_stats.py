import re


def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "stats")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store(cli, skv):
    result = cli("--store=cubi", "my_cube", "stats")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "stats")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_simple(cli, built_cube, skv, store):
    result = cli("--store=cubes", "my_cube", "stats")
    assert result.exit_code == 0

    matcher = re.compile(
        r"""enrich
blobsize:  [1-9]+,[0-9]{3}
files:  2
partitions:  2
rows:  2

source
blobsize:  [1-9]+,[0-9]{3}
files:  2
partitions:  2
rows:  4

__total__
blobsize:  [1-9]+,[0-9]{3}
files:  4"""
    )

    assert matcher.findall(result.output)
