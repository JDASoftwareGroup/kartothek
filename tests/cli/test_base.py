from subprocess import check_call

import pytest
from dask.callbacks import Callback


def test_entry_point(cli):
    check_call("kartothek_cube")


def test_noop(cli):
    result = cli()
    assert result.exit_code == 0
    assert result.output.startswith("Usage: cli")


@pytest.mark.parametrize("arg", ["--help", "-h"])
def test_help(cli, arg):
    result = cli(arg)
    assert result.exit_code == 0
    assert result.output.startswith("Usage: cli")


def test_missing_command(cli):
    result = cli("my_cube")
    assert result.exit_code == 2
    assert "Error: Missing command." in result.output


def test_unknown_command(cli):
    result = cli("my_cube", "foo")
    assert result.exit_code == 2
    assert (
        'Error: No such command "foo".' in result.output
        or "Error: No such command 'foo'." in result.output
    )


def test_cleanup(cli, built_cube, skv):
    # test that interpreter is clean after CLI exists
    cli("--store=cubes", "my_cube", "info")
    assert Callback.active == set()
