"""
The condition sublanguage.
"""
import copy
import itertools
import re
import typing
from collections import defaultdict

import attr
import pandas as pd

from kartothek.serialization import filter_df_from_predicates
from kartothek.utils.converters import (
    converter_str,
    converter_tuple,
    get_str_to_python_converter,
)

__all__ = (
    "C",
    "Condition",
    "Conjunction",
    "EqualityCondition",
    "GreaterEqualCondition",
    "GreaterThanCondition",
    "InIntervalCondition",
    "InequalityCondition",
    "IsInCondition",
    "LessEqualCondition",
    "LessThanCondition",
    "SimpleCondition",
    "VirtualColumn",
)


def _validator_value(instance, attribute, value):
    if pd.isnull(value):
        raise ValueError(
            'Cannot use NULL-value to compare w/ column "{}"'.format(instance.column)
        )
    if isinstance(value, VirtualColumn):
        raise TypeError("Cannot compare two columns.")
    if isinstance(value, (Condition, Conjunction)):
        raise TypeError("Cannot use nested conditions.")


def _validator_valuelist(instance, attribute, value):
    for v in value:
        _validator_value(instance, attribute, v)


def _validator_condlist(instance, attribute, value):
    if any(not isinstance(x, Condition) for x in value):
        raise TypeError("Can only build conjunction out of conditions.")


def _converter_condlist(obj) -> tuple:
    if isinstance(obj, Conjunction):
        return obj.conditions
    elif obj is None:
        return ()
    else:
        return converter_tuple(obj)


@attr.s(frozen=True, eq=False)
class VirtualColumn:
    """
    Virtual column that can be used to easily construct conditions.

    The following operations are supported:

    +---------------+--------------------------------------+-----------------------------------+
    | Operation     | Python Example                       | Result Class                      |
    +===============+======================================+===================================+
    | Equal         | ``C("a") == 42``                     | :py:class:`EqualityCondition`     |
    +---------------+--------------------------------------+-----------------------------------+
    | Not Equal     | ``C("a") != 42``                     | :py:class:`InequalityCondition`   |
    +---------------+--------------------------------------+-----------------------------------+
    | Less Than     | ``C("a") < 42``                      | :py:class:`LessThanCondition`     |
    +---------------+--------------------------------------+-----------------------------------+
    | Less Equal    | ``C("a") <= 42``                     | :py:class:`LessEqualCondition`    |
    +---------------+--------------------------------------+-----------------------------------+
    | Greater Than  | ``C("a") > 42``                      | :py:class:`GreaterThanCondition`  |
    +---------------+--------------------------------------+-----------------------------------+
    | Greater Equal | ``C("a") >= 42``                     | :py:class:`GreaterEqualCondition` |
    +---------------+--------------------------------------+-----------------------------------+
    | Is In         | ``C("a").isin([1, 2])``              | :py:class:`IsInCondition`         |
    +---------------+--------------------------------------+-----------------------------------+
    | In Interval   | ``C("a").in_interval(0, 100)``       | :py:class:`InIntervalCondition`   |
    +---------------+--------------------------------------+-----------------------------------+

    Parameters
    ----------
    name: str
        Column name.
    """

    name = attr.ib(converter=converter_str, type=str)

    def __eq__(self, other):
        return EqualityCondition(self.name, other)

    def __ne__(self, other):
        return InequalityCondition(self.name, other)

    def __lt__(self, other):
        return LessThanCondition(self.name, other)

    def __le__(self, other):
        return LessEqualCondition(self.name, other)

    def __gt__(self, other):
        return GreaterThanCondition(self.name, other)

    def __ge__(self, other):
        return GreaterEqualCondition(self.name, other)

    def isin(self, other):
        return IsInCondition(self.name, other)

    def in_interval(self, start=None, stop=None):
        return InIntervalCondition(self.name, start, stop)


C = VirtualColumn


@attr.s(frozen=True)
class Condition:
    """
    An abstract condition on a column.

    Multiple conditions may be combined using ``&``::

        (C('a') == 1) & (C('b') == 2)

    Parameters
    ----------
    column: str
        Column name.
    """

    column = attr.ib(converter=converter_str, type=str)

    def __bool__(self):
        raise TypeError(
            "Cannot check if a condition is non-zero.\n"
            "Hint: Did you just tried something like `bool(condition)` or `A <= column < B`?"
        )

    __nonzero__ = __bool__  # Python 2

    def __and__(self, other):
        return Conjunction.from_two(self, other)

    def filter_df(self, df):
        """
        Filter given DataFrame w/ condition.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame to evaluate on, must contain required column.

        Returns
        -------
        result: pandas.DataFrame
            Part of the DataFrame for which the condition holds.
        """
        return Conjunction([self]).filter_df(df)

    @staticmethod
    def from_string(s, all_types):
        """
        Parse string as condition object.

        Parameters
        ----------
        s: str
            String to parse.
        all_types: Dict[str, pyarrow.DataType]
            Mapping from all known columns to pyarrow types.

        Returns
        -------
        condition: Condition
            Parsed condition.

        Raises
        ------
        ValueError: If condition cannot be parsed.
        """
        m = re.match(
            pattern=r"""
            ^                    # anchor
            \s*                  # optional space
            \(?                  # optional open bracket
            \s*                  # optional space
            ([^!<>=\s]+)         # column name
            \s*                  # optional space
            (==|=|<=|<|>=|>|!=)  # operator
            ([^)=]+)             # value
            \)?                  # optional closing bracket
            \s*                  # optional space
            $                    # anchor
            """,
            string=s,
            flags=re.VERBOSE,
        )
        if not m:
            raise ValueError('Cannot parse condition "{s}"'.format(s=s))
        col, op, var = m.groups()

        col_obj = C(col)

        pa_type = all_types.get(col)
        if pa_type is None:
            raise ValueError(
                'Unknown column "{col}" in condition "{s}"'.format(col=col, s=s)
            )
        var_f = get_str_to_python_converter(pa_type)
        var_obj = var_f(var.strip())

        if (op == "==") or (op == "="):
            return col_obj == var_obj
        elif op == "<=":
            return col_obj <= var_obj
        elif op == "<":
            return col_obj < var_obj
        elif op == ">=":
            return col_obj >= var_obj
        elif op == ">":
            return col_obj > var_obj
        elif op == "!=":
            return col_obj != var_obj
        else:
            raise RuntimeError("unreachable")


@attr.s(frozen=True)
class SimpleCondition(Condition):
    """
    A simple condition that only emits a single predicate part. Must be subclassed.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    value = attr.ib(validator=[_validator_value])
    active = True

    def __str__(self):
        return "{column} {op} {value}".format(
            column=self.column, op=self.OP, value=self.value
        )

    @property
    def predicate_part(self):
        """
        Part of the inner list for Kartothek predicate pushdown.
        """
        return [(self.column, self.OP, self.value)]


@attr.s(frozen=True)
class EqualityCondition(SimpleCondition):
    """
    Condition on column equality.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = "=="


@attr.s(frozen=True)
class InequalityCondition(SimpleCondition):
    """
    Condition on column inequality.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = "!="


@attr.s(frozen=True)
class LessThanCondition(SimpleCondition):
    """
    Condition that describes that a column should be strictly less than the given value.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = "<"


@attr.s(frozen=True)
class LessEqualCondition(SimpleCondition):
    """
    Condition that describes that a column should be less or equal to the given value.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = "<="


@attr.s(frozen=True)
class GreaterThanCondition(SimpleCondition):
    """
    Condition that describes that a column should be strictly greater than the given value.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = ">"


@attr.s(frozen=True)
class GreaterEqualCondition(SimpleCondition):
    """
    Condition that describes that a column should be greater or equal to the given value.

    Parameters
    ----------
    column: str
        Column name.
    value: Any
        To which value the column should be compared to.
    """

    OP = ">="


@attr.s(frozen=True)
class IsInCondition(SimpleCondition):
    """
    Condition that describes that values in a column should be within the given list.

    Parameters
    ----------
    columns: str
        Column name.
    value: Tuple[Any]
        Tuple to check for.
    """

    OP = "in"

    value = attr.ib(
        converter=converter_tuple,
        type=typing.Tuple[typing.Any],
        validator=[_validator_valuelist],
    )


@attr.s(frozen=True)
class InIntervalCondition(Condition):
    """
    Condition expressing that values of a column should be in a given interval.

    Parameters
    ----------
    columns: str
        Column name.
    start: Any
        Inclusive start of the interval, optional.
    stop: Any
        Exclusive stop of the interval, optional.
    """

    start = attr.ib(
        default=None, validator=[attr.validators.optional(_validator_value)]
    )
    stop = attr.ib(default=None, validator=[attr.validators.optional(_validator_value)])

    def __str__(self):
        return "{column}.in_interval({start}, {stop})".format(
            column=self.column, start=self.start, stop=self.stop
        )

    @property
    def predicate_part(self):
        """
        Part of the inner list for Kartothek predicate pushdown.
        """
        result = []
        if self.start is not None:
            result.append((self.column, ">=", self.start))
        if self.stop is not None:
            result.append((self.column, "<", self.stop))
        return result

    @property
    def active(self):
        return (self.start is not None) or (self.stop is not None)


@attr.s(frozen=True)
class Conjunction:
    """
    Conjunction of multiple :class:`Condition` objects.

    Parameters
    ----------
    conditions: Tuple[Condition]
        Tuple of conditions that must all be satisfied at the same time. Can address multiple columns.
    """

    conditions = attr.ib(
        converter=_converter_condlist,
        type=typing.Tuple[Condition],
        validator=[_validator_condlist],
    )

    @classmethod
    def from_two(cls, left, right):
        """
        Create conjunction from two elements.

        Parameters
        ----------
        left: Union[Condition, Conjunction]:
            Left part.
        right: Union[Condition, Conjunction]:
            Right part.

        Returns
        -------
        conjunction: Conjunction
            Conjunction of the two given parts.
        """
        conditions = []

        for obj in (left, right):
            if isinstance(obj, Conjunction):
                conditions += obj.conditions
            else:
                conditions.append(obj)

        return cls(conditions)

    def __and__(self, other):
        return Conjunction.from_two(self, other)

    def __str__(self):
        return " & ".join("({})".format(cond) for cond in self.conditions)

    @property
    def columns(self):
        """
        Columns that are checked by this conjunction.
        """
        return {cond.column for cond in self.conditions if cond.active}

    @property
    def predicate(self):
        """
        Predicate to be consumed by Kartothek and DataFrame serializer.
        """
        result = list(
            itertools.chain.from_iterable(
                cond.predicate_part for cond in self.conditions
            )
        )
        if result:
            return result
        else:
            return None

    def split_by_column(self):
        """
        Split conjunction by column.

        Non-active conditions will be dropped.

        Returns
        -------
        split: Dict[str, Conjunction]
            Conjunctions by affected column.
        """
        parts = defaultdict(list)
        for cond in self.conditions:
            if cond.active:
                parts[cond.column].append(cond)
        return {column: Conjunction(part) for column, part in parts.items()}

    def filter_df(self, df):
        """
        Filter given DataFrame w/ conjunction.

        NULL-values will always treated as non-matching.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame to evaluate on, must contain required column.

        Returns
        -------
        result: pandas.DataFrame
            Part of the DataFrame for which the conjunction holds.
        """
        df = df.loc[df[list(self.columns)].notnull().all(axis=1)]

        predicate = self.predicate
        if predicate is None:
            # kartothek does not support empty predicate lists
            return df
        else:
            return filter_df_from_predicates(df, [self.predicate])

    def to_jsonarray(self):
        """
        Converts conjunction to a list that can be used for JSON/YAML serialization.

        .. important::
            Not all value types that can be used within conditions are JSON-serializable (e.g. ``datetime`` objects).
            The user is responsible of ensuring that these values can pass functions like ``json.dump`` or has to
            implement proper error handling.

        Returns
        -------
        jsonarray: List[Dict[str, Any]]
            JSON-compatible array.

        Example
        -------
        >>> import json
        >>> from kartothek.core.cube.conditions import C
        >>> conjunction = (
        ...     (C("x") > 1)
        ...     & (C("y").isin(["foo", "bar"]))
        ... )
        >>> array = conjunction.to_jsonarray()
        >>> print(json.dumps(array, indent=True, sort_keys=True))
        [
         {
          "column": "x",
          "type": "GreaterThanCondition",
          "value": 1
         },
         {
          "column": "y",
          "type": "IsInCondition",
          "value": [
           "foo",
           "bar"
          ]
         }
        ]

        See Also
        --------
        from_jsonarray: Converts array back into a conjunction.
        """
        jsonarray = []
        for cond in self.conditions:
            d = attr.asdict(cond)
            d["type"] = type(cond).__name__
            jsonarray.append(d)
        return jsonarray

    @staticmethod
    def from_jsonarray(array):
        """
        Recover conjunction from JSON-compatible array.

        Parameters
        ----------
        jsonarray: List[Dict[str, Any]]
            JSON-compatible array.

        Returns
        -------
        conjunction: Conjunction
            Recovered conjunction.

        Raises
        ------
        TypeError: If are wrong or unknown condition type was passed.
        ValueError: If ``"type"`` attribute within a condition is missing.

        See Also
        --------
        to_jsonarray: Creates array, illustrates format.
        """
        if not isinstance(array, list):
            raise TypeError("jsonarray must be a list")

        # find all possible classes
        all_classes = {}
        seen = set()
        todo = [Condition]
        for c in todo:
            if c in seen:
                continue
            sub = c.__subclasses__()
            if sub:
                # not a leaf
                todo += c.__subclasses__()
            else:
                # leaf == found a class
                all_classes[c.__name__] = c
            seen.add(c)

        # deserialize all conditions
        conditions = []
        for element in array:
            if not isinstance(element, dict):
                raise TypeError("Condition in jsonarray must be a dict")

            element = copy.deepcopy(element)
            if "type" not in element:
                raise ValueError("Missing type value for condition")

            t = element.pop("type")
            if t not in all_classes:
                raise TypeError(f"Unknown condition class '{t}'")

            c = all_classes[t]
            conditions.append(c(**element))

        return Conjunction(conditions)

    @staticmethod
    def from_string(s, all_types):
        """
        Parse string as conjunction object.

        .. important::
            This is intended to be used for human interaction (e.g. CLIs). Do not use this for serializing and
            deserializing conditions, since this does not support all conditions and is not guaranteed to be
            roundtrip-safe. For the purpose of serialization, better use :meth:`to_jsonarray` and
            :meth:`from_jsonarray`.

        Parameters
        ----------
        s: str
            String to parse.
        all_types: Dict[str, pyarrow.DataType]
            Mapping from all known columns to pyarrow types.

        Returns
        -------
        conjunction: Conjunction
            Parsed conjunction.

        Raises
        ------
        ValueError: If condition cannot be parsed.
        """
        s = s.strip()
        if s:
            return Conjunction(
                [Condition.from_string(sub, all_types) for sub in s.split("&")]
            )
        else:
            return Conjunction([])
