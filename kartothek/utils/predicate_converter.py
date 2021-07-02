"""
Helper module to convert kartothek dataset load predicates into cube conditions.
"""

from typing import Any, List, Sequence, Tuple

import pandas as pd
import pyarrow as pa

from kartothek.core.cube.conditions import Condition


def write_predicate_as_cube_condition(predicate: Tuple[str, str, Any]) -> Condition:
    """
    Rewrites a single io.dask.dataset 'read_dataset_as_ddf' predicate condition as cube condition.

    Remark: This function is restricted by "Condition.from_string" which does not allow for IsInCondition
    and InIntervalCondition and will throw an error if conditions of those types are passed.

    Parameters
    ----------
    predicate: list
        list containing single predicate definition

    Returns
    -------
    condition: Condition
        cube condition containing the predicate definition
    """
    condition_string = None
    parameter_format_dict = {}

    if len(predicate) != 3:
        raise ValueError("Please use predicates consisting of exactly 3 entries")

    if type(predicate[2]) == int:
        condition_string = f"{predicate[0]} {predicate[1]} {str(predicate[2])}"
        parameter_format_dict[predicate[0]] = pa.int64()
    if type(predicate[2]) == str:
        condition_string = f"{predicate[0]} {predicate[1]} {predicate[2]}"
        parameter_format_dict[predicate[0]] = pa.string()
    if type(predicate[2]) == pd._libs.tslibs.timestamps.Timestamp:
        condition_string = (
            f"{predicate[0]} {predicate[1]} {predicate[2].strftime('%Y-%m-%d')}"
        )
        parameter_format_dict[predicate[0]] = pa.timestamp("s")
    if type(predicate[2]) == bool:
        condition_string = f"{predicate[0]} {predicate[1]} {str(predicate[2])}"
        parameter_format_dict[predicate[0]] = pa.bool_()
    if type(predicate[2]) == float:
        condition_string = f"{predicate[0]} {predicate[1]} {str(predicate[2])}"
        parameter_format_dict[predicate[0]] = pa.float64()

    if condition_string is not None:
        condition = Condition.from_string(condition_string, parameter_format_dict)
    else:
        raise TypeError(
            "Please only enter predicates for parameter values of the following type:"
            " str, int, float, bool or pandas timestamp, "
        )
    return condition


def convert_predicates_to_cube_conditions(
    predicates: List[List[Tuple[str, str, Any]]],
) -> Sequence[Condition]:
    """
    Converts a io.dask.dataset 'read_dataset_as_ddf' predicate to a cube condition

    Parameters
    ----------
    predicates: list
        list containing a list of single predicates

    Returns
    -------
    condition: Condition
        cube condition containing the combined predicate definitions
    """
    condition: Any = ()
    if len(predicates) > 1:
        raise ValueError(
            "Cube conditions cannot handle 'or' operators, therefore, "
            "please pass a predicate list with one element."
        )
    for predicate in predicates[0]:
        condition = condition + (write_predicate_as_cube_condition(predicate),)
    return condition
