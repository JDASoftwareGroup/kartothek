"""
Helper module to convert user inputs into normalized forms.
"""
from __future__ import absolute_import

from typing import Iterable, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
from kartothek.core.cube.conditions import Condition

__all__ = (
    "converter_str",
    "converter_str_set",
    "converter_str_set_optional",
    "converter_str_tupleset",
    "converter_tuple",
    "get_str_to_python_converter",
)


def converter_str_set(obj) -> frozenset:
    """
    Convert input to a set of unicode strings. ``None`` will be converted to an empty set.

    Parameters
    ----------
    obj: Optional[Union[Iterable[str], str]]
        Object to convert.

    Returns
    -------
    obj: FrozenSet[str]
        String set.

    Raises
    ------
    TypeError
        If passed object is not string/byte-like.
    """
    result = converter_tuple(obj)
    result_set = {converter_str(x) for x in result}
    return frozenset(result_set)


def converter_str_set_optional(obj):
    """
    Convert input to a set of unicode strings. ``None`` will be preserved.

    Parameters
    ----------
    obj: Optional[Union[Iterable[str], str]]
        Object to convert.

    Returns
    -------
    obj: Optional[FrozenSet[str]]
        String set.

    Raises
    ------
    ValueError
        If an element in the passed object is not string/byte/like.
    """
    if obj is None:
        return None
    return converter_str_set(obj)


def converter_str_tupleset(obj: Optional[Union[Iterable[str], str]]) -> Tuple[str, ...]:
    """
    Convert input to tuple of unique unicode strings. ``None`` will be converted to an empty set.

    The input must not contain duplicate entries.

    Parameters
    ----------
    obj
        Object to convert.

    Raises
    ------
    TypeError
        If passed object is not string/byte-like, or if ``obj`` is known to have an unstable iteration order.
    ValueError
        If passed set contains duplicates.
    """
    if isinstance(obj, (dict, frozenset, set)):
        raise TypeError(
            "{obj} which has type {tname} has an unstable iteration order".format(
                obj=obj, tname=type(obj).__name__
            )
        )
    result = converter_tuple(obj)
    result = tuple(converter_str(x) for x in result)
    if len(set(result)) != len(result):
        raise ValueError("Tuple-set contains duplicates: {}".format(", ".join(result)))
    return result


def converter_tuple(obj) -> tuple:
    """
    Convert input to a tuple. ``None`` will be converted to an empty tuple.

    Parameters
    ----------
    obj: Any
        Object to convert.

    Returns
    -------
    obj: Tuple[Any]
        Tuple.
    """
    if obj is None:
        return ()
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return tuple(x for x in obj)
    else:
        return (obj,)


def converter_str(obj) -> str:
    """
    Ensures input is a unicode string.

    Parameters
    ----------
    obj: str
        Object to convert.

    Returns
    -------
    obj: str
        String.

    Raises
    ------
    TypeError
        If passed object is not string/byte-like.
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    else:
        raise TypeError(
            "Object of type {type} is not a string: {obj}".format(
                obj=obj, type=type(obj).__name__
            )
        )


def get_str_to_python_converter(pa_type):
    """
    Get converter to parse string into python object.

    Parameters
    ----------
    pa_type: pyarrow.DataType
        Data type.

    Returns
    -------
    converter: Callable[[str], Any]
        Converter.
    """
    if pa.types.is_boolean(pa_type):

        def var_f(x):
            if x.lower() in ("0", "f", "n", "false", "no"):
                return False
            elif x.lower() in ("1", "t", "y", "true", "yes"):
                return True
            else:
                raise ValueError("Cannot parse bool: {}".format(x))

        return var_f
    elif pa.types.is_floating(pa_type):
        return float
    elif pa.types.is_integer(pa_type):
        return int
    elif pa.types.is_string(pa_type):

        def var_f(x):
            if len(x) > 1:
                for char in ('"', "'"):
                    if x.startswith(char) and x.endswith(char):
                        return x[1:-1]
            return x

        return var_f
    elif pa.types.is_timestamp(pa_type):
        return pd.Timestamp
    else:
        raise ValueError("Cannot handle type {pa_type}".format(pa_type=pa_type))


def write_predicate_as_cube_condition(predicate):
    """
    Rewrites a single io.dask.dataset 'read_dataset_as_ddf' predicate condition as cube condition

    Parameters
    ----------
    predicate_list: list
        list containing single predicate definition

    Returns
    -------
    condition: cube condition object
        cube condition containing the predicate definition
    """
    parameter_format_dict = {}
    if type(predicate[2]) == int:
        condition_string = '{} {} {}'.format(predicate[0], predicate[1], str(predicate[2]))
        parameter_format_dict[predicate[0]] = pa.int16()
    if type(predicate[2]) == str:
        condition_string = '{} {} {}'.format(predicate[0], predicate[1], predicate[2])
        parameter_format_dict[predicate[0]] = pa.str()
    if type(predicate[2]) == pd._libs.tslibs.timestamps.Timestamp:
        condition_string = '{} {} {}'.format(predicate[0], predicate[1], predicate[2].strftime('%Y-%m-%d'))
        parameter_format_dict[predicate[0]] = pa.timestamp('s')
    if type(predicate[2]) == bool:
        condition_string = '{} {} {}'.format(predicate[0], predicate[1], str(predicate[2]))
        parameter_format_dict[predicate[0]] = pa.bool_()
    if type(predicate[2]) == float:
        condition_string = '{} {} {}'.format(predicate[0], predicate[1], str(predicate[2]))
        parameter_format_dict[predicate[0]] = pa.float64()

    if condition_string is not None:
        condition = Condition.from_string(condition_string, parameter_format_dict)
    else:
        raise TypeError(
            "Please enter only enter predicates for parameter values of the following type:"
            " str, int, float, bool or pandas timestamp, "
        )
    return condition


def convert_predicates_to_cube_conditions(predicates):
    """
    Converts a io.dask.dataset 'read_dataset_as_ddf' predicate to a cube condition

    Parameters
    ----------
    predicates: list
        list containing a list of single predicates

    Returns
    -------
    condition: cube condition object
        cube condition containing the combined predicate definitions
    """
    condition = ()
    for predicate in predicates[0]:
        condition = condition + (write_predicate_as_cube_condition(predicate),)
    return condition
