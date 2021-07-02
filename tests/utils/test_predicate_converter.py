import pandas as pd
import pytest

from kartothek.core.cube.conditions import C
from kartothek.utils.predicate_converter import (
    convert_predicates_to_cube_conditions,
    write_predicate_as_cube_condition,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (["column", "==", 1], (C("column") == 1)),
        (["column", "==", 1.1], C("column") == 1.1),
        (["column", "==", "1"], C("column") == "1"),
        (
            ["date", "==", pd.to_datetime("2020-01-01")],
            (C("date") == pd.Timestamp("2020-01-01 00:00:00")),
        ),
        (["column", "==", True], C("column") == True),  # noqa: E712)
        (["column", "<=", 1], (C("column") <= 1)),
        (["column", ">=", 1], (C("column") >= 1)),
        (["column", "!=", 1], (C("column") != 1)),
    ],
)
def test_write_predicate_as_cube_condition(test_input, expected):
    assert write_predicate_as_cube_condition(test_input) == expected


def test_raises_type_error_write_predicate_as_cube_condition():
    with pytest.raises(
        TypeError,
        match="Please only enter predicates for parameter values of the "
        "following type: str, int, float, bool or pandas timestamp, ",
    ):
        write_predicate_as_cube_condition(
            ("date", "==", pd.to_datetime("2020-01-01").date())
        )


def test_raises_value_error_write_predicate_as_cube_condition():
    with pytest.raises(
        ValueError, match="Please use predicates consisting of exactly 3 entries"
    ):
        write_predicate_as_cube_condition(("date", "=="))

    with pytest.raises(
        ValueError, match="Please use predicates consisting of exactly 3 entries"
    ):
        write_predicate_as_cube_condition(("date", "==", "date", "=="))


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[("column", "==", 1)]], (C("column") == 1,)),
        (
            [[("column", "==", 1), ("column2", "==", "1")]],
            (C("column") == 1, C("column2") == "1"),
        ),
    ],
)
def test_convert_predicates_to_cube_conditions(test_input, expected):
    assert convert_predicates_to_cube_conditions(test_input) == expected


def test_raises_value_error_convert_predicates_to_cube_conditions():
    with pytest.raises(
        ValueError,
        match="Cube conditions cannot handle 'or' operators, therefore, "
        "please pass a predicate list with one element.",
    ):
        convert_predicates_to_cube_conditions(
            [[("column", "==", 1)], [("column2", "==", 2)]]
        )
