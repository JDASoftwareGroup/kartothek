import builtins
from copy import copy

import pytest


class _PromptPatch:
    def __init__(self, history):
        self.history = history
        self.args = []

    def __call__(self, *args, **kwargs):
        self.args.append((args, kwargs))

        if self.history:
            return self.history.pop(0)
        else:
            raise KeyboardInterrupt()


@pytest.fixture
def mock_prompt(mocker):
    def _f(history):
        history = copy(history)
        patch = _PromptPatch(history)
        mocker.patch("kartothek.cli._query.prompt", patch)
        return patch

    return _f


def test_no_ipython(cli, built_cube, mock_prompt, mocker, skv):
    mock_prompt([])

    realimport = builtins.__import__

    def myimport(name, *args, **kwargs):
        if name == "IPython":
            raise ImportError("Module not found.")
        return realimport(name, *args, **kwargs)

    try:
        builtins.__import__ = myimport
        result = cli("--store=cubes", "my_cube", "query")
        assert result.exit_code == 2
        assert "Error: Could not load IPython" in result.output
    finally:
        builtins.__import__ = realimport


def test_no_input(cli, built_cube, mock_prompt, skv):
    mock_prompt([])
    result = cli("--store=cubes", "my_cube", "query")
    assert result.exit_code == 1
    assert result.output == "\nAborted!\n"


def test_simple(cli, built_cube, mock_prompt, df_complete, skv):
    mock_prompt(["", ""])  # conditions  # payload
    result = cli("--store=cubes", "my_cube", "query", input="df\n")

    str_df = str(df_complete.loc[:, ["p", "q", "x", "y"]].reset_index(drop=True))

    assert result.exit_code == 1
    assert str_df in result.output


def test_payload_select(cli, built_cube, mock_prompt, df_complete, skv):
    mock_prompt(["", "v1,v2"])  # conditions  # payload
    result = cli("--store=cubes", "my_cube", "query", input="df\n")

    str_df = str(
        df_complete.loc[:, ["p", "q", "v1", "v2", "x", "y"]].reset_index(drop=True)
    )

    assert result.exit_code == 1
    assert str_df in result.output


def test_payload_all(cli, built_cube, mock_prompt, df_complete, skv):
    mock_prompt(["", "__all__"])  # conditions  # payload
    result = cli("--store=cubes", "my_cube", "query", input="df\n")

    str_df = str(df_complete.loc[:, :].reset_index(drop=True))

    assert result.exit_code == 1
    assert str_df in result.output


def test_payload_all_roundtrip(cli, built_cube, mock_prompt, skv):
    """
    There was a bug that resulted in __all__ expanded to the list of all columns for the second promp round, which is
    not nice if the list of cube columns is very long.
    """
    prompt = mock_prompt(
        ["", "__all__", "", ""]  # conditions  # payload  # conditions  # payload
    )
    result = cli("--store=cubes", "my_cube", "query", input="exit()\n")

    assert result.exit_code == 1
    assert (
        len(prompt.args) == 5
    )  # 2 successfull rounds w/ 2 prompts each and 1 interrupted

    # test default arg of second round payload
    args, kwargs = prompt.args[3]
    assert kwargs["default"] == "__all__"


def test_conditions(cli, built_cube, mock_prompt, df_complete, skv):
    mock_prompt(["(i1 == False) & (v1 > 20)", ""])  # conditions  # payload
    result = cli("--store=cubes", "my_cube", "query", input="df\n")

    str_df = str(
        df_complete.loc[
            (df_complete["i1"] == False) & (df_complete["v1"] > 20.0),  # noqa
            ["p", "q", "x", "y"],
        ].reset_index(drop=True)
    )

    assert result.exit_code == 1
    assert str_df in result.output
