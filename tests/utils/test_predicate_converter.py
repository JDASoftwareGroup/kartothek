import pandas as pd
import pytest

from kartothek.core.cube.conditions import C
from kartothek.utils.predicate_converter import (
    convert_predicates_to_cube_conditions,
    write_predicate_as_cube_condition,
)


def test_write_predicate_as_cube_condition():
    assert write_predicate_as_cube_condition(["column", "==", 1]) == (C("column") == 1)
    assert write_predicate_as_cube_condition(["column", "==", 1.1]) == (
        C("column") == 1.1
    )
    assert write_predicate_as_cube_condition(["column", "==", "1"]) == (
        C("column") == "1"
    )
    assert write_predicate_as_cube_condition(
        ["date", "==", pd.to_datetime("2020-01-01")]
    ) == (C("date") == pd.Timestamp("2020-01-01 00:00:00"))
    assert write_predicate_as_cube_condition(["column", "==", True]) == (
        C("column") == True  # noqa: E712
    )

    assert write_predicate_as_cube_condition(["column", "<=", 1]) == (C("column") <= 1)
    assert write_predicate_as_cube_condition(["column", ">=", 1]) == (C("column") >= 1)
    assert write_predicate_as_cube_condition(["column", "!=", 1]) == (C("column") != 1)

    with pytest.raises(
        TypeError,
        match="Please only enter predicates for parameter values of the "
        "following type: str, int, float, bool or pandas timestamp, ",
    ):
        write_predicate_as_cube_condition(
            ("date", "==", pd.to_datetime("2020-01-01").date())
        )

    with pytest.raises(
        ValueError, match="Please use predicates consisting of exactly 3 entries"
    ):
        write_predicate_as_cube_condition(("date", "=="))

    with pytest.raises(
        ValueError, match="Please use predicates consisting of exactly 3 entries"
    ):
        write_predicate_as_cube_condition(("date", "==", "date", "=="))


def test_convert_predicates_to_cube_conditions():
    assert convert_predicates_to_cube_conditions([[("column", "==", 1)]]) == (
        C("column") == 1,
    )
    assert convert_predicates_to_cube_conditions(
        [[("column", "==", 1), ("column2", "==", "1")]]
    ) == (C("column") == 1, C("column2") == "1")

    with pytest.raises(
        ValueError,
        match="Cube conditions cannot handle 'or' operators, therefore, "
        "please pass a predicate list with one element.",
    ):
        convert_predicates_to_cube_conditions(
            [[("column", "==", 1)], [("column2", "==", 2)]]
        )
