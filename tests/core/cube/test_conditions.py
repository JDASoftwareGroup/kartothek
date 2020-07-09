# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

from kartothek.core.cube.conditions import (
    C,
    Condition,
    Conjunction,
    EqualityCondition,
    GreaterEqualCondition,
    GreaterThanCondition,
    InequalityCondition,
    InIntervalCondition,
    IsInCondition,
    LessEqualCondition,
    LessThanCondition,
)


class TestVirtualColumn:
    def test_convert(self):
        c = C(b"foo")
        assert c.name == "foo"
        assert isinstance(c.name, str)

    def test_frozen(self):
        c = C("foo")
        with pytest.raises(AttributeError):
            c.name = "bar"


class TestSimpleCondition:
    @pytest.mark.parametrize(
        "f,t,op,value",
        [
            (
                # f
                lambda c: c == 42,
                # t
                EqualityCondition,
                # op
                "==",
                # value
                42,
            ),
            (
                # f
                lambda c: c != 42,
                # t
                InequalityCondition,
                # op
                "!=",
                # value
                42,
            ),
            (
                # f
                lambda c: c < 42,
                # t
                LessThanCondition,
                # op
                "<",
                # value
                42,
            ),
            (
                # f
                lambda c: c <= 42,
                # t
                LessEqualCondition,
                # op
                "<=",
                # value
                42,
            ),
            (
                # f
                lambda c: c > 42,
                # t
                GreaterThanCondition,
                # op
                ">",
                # value
                42,
            ),
            (
                # f
                lambda c: c >= 42,
                # t
                GreaterEqualCondition,
                # op
                ">=",
                # value
                42,
            ),
            (
                # f
                lambda c: c.isin(42),
                # t
                IsInCondition,
                # op
                "in",
                # value
                (42,),
            ),
        ],
    )
    def test_op(self, f, t, op, value):
        c = C("foö")
        cond = f(c)
        assert isinstance(cond, t)
        assert cond.OP == op
        assert str(cond) == "foö {} {}".format(op, value)
        assert cond.predicate_part == [("foö", op, value)]
        assert cond.active
        hash(cond)

    def test_frozen(self):
        cond = C("foö") == 42

        with pytest.raises(AttributeError):
            cond.column = "bar"

        with pytest.raises(AttributeError):
            cond.value = 1337

        with pytest.raises(AttributeError):
            cond.OP = "x"

    def test_filter_df(self):
        cond = C("foö") == 42
        df = pd.DataFrame({"foö": [13, 42, 42, 100], "bar": 0.0})
        df_actual = cond.filter_df(df)
        df_expected = df.loc[df["foö"] == 42]
        pdt.assert_frame_equal(df_actual, df_expected)

    def test_fails_null_scalar(self):
        with pytest.raises(ValueError) as exc:
            C("foö") == None  # noqa
        assert str(exc.value) == 'Cannot use NULL-value to compare w/ column "foö"'

    def test_fails_null_list(self):
        with pytest.raises(ValueError) as exc:
            C("foö").isin([0, None, 1])
        assert str(exc.value) == 'Cannot use NULL-value to compare w/ column "foö"'

    def test_fails_colcol_scalar(self):
        c1 = C("foö")
        c2 = C("bar")
        with pytest.raises(TypeError) as exc:
            c1 == c2
        assert str(exc.value) == "Cannot compare two columns."

    def test_fails_colcol_list(self):
        c1 = C("foö")
        c2 = C("bar")
        with pytest.raises(TypeError) as exc:
            c1.isin([c2])
        assert str(exc.value) == "Cannot compare two columns."

    def test_fails_colcond_scalar(self):
        c1 = C("foö")
        c2 = C("bar")
        cond = c2 == 42
        with pytest.raises(TypeError) as exc:
            c1 == cond
        assert str(exc.value) == "Cannot use nested conditions."

    def test_fails_colcond_list(self):
        c1 = C("foö")
        c2 = C("bar")
        cond = c2 == 42
        with pytest.raises(TypeError) as exc:
            c1.isin([cond])
        assert str(exc.value) == "Cannot use nested conditions."

    def test_fails_colconj_scalar(self):
        c1 = C("foö")
        c2 = C("bar")
        conj = (c2 == 42) & (c2 == 10)
        with pytest.raises(TypeError) as exc:
            c1 == conj
        assert str(exc.value) == "Cannot use nested conditions."

    def test_fails_colconj_list(self):
        c1 = C("foö")
        c2 = C("bar")
        conj = (c2 == 42) & (c2 == 10)
        with pytest.raises(TypeError) as exc:
            c1.isin([conj])
        assert str(exc.value) == "Cannot use nested conditions."

    def test_fails_doublecompare(self):
        with pytest.raises(TypeError) as exc:
            1 < C("foö") <= 5
        assert str(exc.value).startswith("Cannot check if a condition is non-zero.")

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("sö == a", C("sö") == "a"),
            ("sö = a", C("sö") == "a"),
            ("sö==a", C("sö") == "a"),
            ("sö=='a b'", C("sö") == "a b"),
            ("iö != 10", C("iö") != 10),
            ("iö > 10", C("iö") > 10),
            ("iö < 10", C("iö") < 10),
            ("iö >= 10", C("iö") >= 10),
            ("iö <= 10", C("iö") <= 10),
            (" sö == a ", C("sö") == "a"),
            ("( sö == a )", C("sö") == "a"),
            ("tö == 2018-01-01", C("tö") == pd.Timestamp("2018-01-01")),
        ],
    )
    def test_from_string_ok(self, s, expected):
        all_types = {
            "sö": pa.string(),
            "bö": pa.bool_(),
            "iö": pa.int16(),
            "tö": pa.timestamp("ns"),
        }
        actual = Condition.from_string(s, all_types)
        assert actual == expected

        s2 = str(actual)
        actual2 = Condition.from_string(s2, all_types)
        assert actual2 == actual

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("zö == a", 'Unknown column "zö" in condition "zö == a"'),
            ("sö ==", 'Cannot parse condition "sö =="'),
            ("== a", 'Cannot parse condition "== a"'),
            ("sö <=", 'Cannot parse condition "sö <="'),
        ],
    )
    def test_from_string_error(self, s, expected):
        all_types = {"sö": pa.string(), "bö": pa.bool_(), "iö": pa.int16()}
        with pytest.raises(ValueError) as exc:
            Condition.from_string(s, all_types)
        assert str(exc.value) == expected


class TestInIntervaCondition:
    def test_simple(self):
        cond = C("foö").in_interval(10, 20)
        assert isinstance(cond, InIntervalCondition)
        assert str(cond) == "foö.in_interval(10, 20)"
        assert cond.predicate_part == [("foö", ">=", 10), ("foö", "<", 20)]
        assert cond.active
        hash(cond)

    def test_begin_null(self):
        cond = C("foö").in_interval(stop=20)
        assert isinstance(cond, InIntervalCondition)
        assert str(cond) == "foö.in_interval(None, 20)"
        assert cond.predicate_part == [("foö", "<", 20)]
        assert cond.active

    def test_end_null(self):
        cond = C("foö").in_interval(10)
        assert isinstance(cond, InIntervalCondition)
        assert str(cond) == "foö.in_interval(10, None)"
        assert cond.predicate_part == [("foö", ">=", 10)]
        assert cond.active

    def test_both_null(self):
        cond = C("foö").in_interval()
        assert isinstance(cond, InIntervalCondition)
        assert str(cond) == "foö.in_interval(None, None)"
        assert cond.predicate_part == []
        assert not cond.active

    def test_fails_null(self):
        col1 = C("foö")
        with pytest.raises(ValueError) as exc:
            col1.in_interval(10, np.nan)
        assert str(exc.value) == 'Cannot use NULL-value to compare w/ column "foö"'

    def test_fails_colcol(self):
        col1 = C("foö")
        col2 = C("bar")
        with pytest.raises(TypeError) as exc:
            col1.in_interval(10, col2)
        assert str(exc.value) == "Cannot compare two columns."

    def test_fails_colcond(self):
        col1 = C("foö")
        col2 = C("bar")
        cond = col2 == 42
        with pytest.raises(TypeError) as exc:
            col1.in_interval(10, cond)
        assert str(exc.value) == "Cannot use nested conditions."

    def test_fails_colconj(self):
        col1 = C("foö")
        col2 = C("bar")
        conj = (col2 == 42) & (col2 == 10)
        with pytest.raises(TypeError) as exc:
            col1.in_interval(10, conj)
        assert str(exc.value) == "Cannot use nested conditions."


class TestConjunction:
    def test_simple(self):
        col = C("foö")
        cond1 = col < 10
        cond2 = col > 0
        conj = cond1 & cond2
        assert isinstance(conj, Conjunction)
        assert conj.conditions == (cond1, cond2)
        assert str(conj) == "(foö < 10) & (foö > 0)"
        assert conj.columns == {"foö"}
        assert conj.predicate == [("foö", "<", 10), ("foö", ">", 0)]
        assert conj.split_by_column() == {"foö": conj}

    def test_nested_conj_cond(self):
        col = C("foö")
        cond1 = col < 10
        cond2 = col > 0
        cond3 = col != 10
        conj1 = cond1 & cond2
        conj2 = conj1 & cond3
        assert isinstance(conj2, Conjunction)
        assert conj2.conditions == (cond1, cond2, cond3)
        assert str(conj2) == "(foö < 10) & (foö > 0) & (foö != 10)"
        assert conj2.columns == {"foö"}
        assert conj2.predicate == [
            ("foö", "<", 10),
            ("foö", ">", 0),
            ("foö", "!=", 10),
        ]
        assert conj2.split_by_column() == {"foö": conj2}

    def test_nested_cond_conj(self):
        col = C("foö")
        cond1 = col < 10
        cond2 = col > 0
        cond3 = col != 10
        conj1 = cond2 & cond3
        conj2 = cond1 & conj1
        assert isinstance(conj2, Conjunction)
        assert conj2.conditions == (cond1, cond2, cond3)

    def test_nested_conj_conj(self):
        col = C("foö")
        cond1 = col < 10
        cond2 = col > 0
        cond3 = col != 10
        cond4 = col != 11
        conj1 = cond1 & cond2
        conj2 = cond3 & cond4
        conj3 = conj1 & conj2
        assert isinstance(conj3, Conjunction)
        assert conj3.conditions == (cond1, cond2, cond3, cond4)

    def test_fails_nocond(self):
        col = C("foö")
        cond1 = col < 10
        with pytest.raises(TypeError) as exc:
            cond1 & col
        assert str(exc.value) == "Can only build conjunction out of conditions."

    def test_multicol(self):
        col1 = C("foö")
        col2 = C("bar")
        cond1 = col1 < 10
        cond2 = col1 > 0
        cond3 = col2 != 10
        conj1 = cond1 & cond2
        conj2 = conj1 & cond3
        assert isinstance(conj2, Conjunction)
        assert conj2.conditions == (cond1, cond2, cond3)
        assert str(conj2) == "(foö < 10) & (foö > 0) & (bar != 10)"
        assert conj2.columns == {"foö", "bar"}
        assert conj2.predicate == [
            ("foö", "<", 10),
            ("foö", ">", 0),
            ("bar", "!=", 10),
        ]
        assert conj2.split_by_column() == {"foö": conj1, "bar": Conjunction([cond3])}

    def test_empty_real(self):
        conj = Conjunction([])
        assert conj.conditions == ()
        assert str(conj) == ""
        assert conj.columns == set()
        assert conj.predicate is None
        assert conj.split_by_column() == {}

    def test_empty_pseudo(self):
        cond = InIntervalCondition("x")
        conj = Conjunction([cond])
        assert conj.conditions == (cond,)
        assert str(conj) == "(x.in_interval(None, None))"
        assert conj.columns == set()
        assert conj.predicate is None
        assert conj.split_by_column() == {}

    def test_filter_df_some(self):
        cond = (C("foö") == 42) & (C("bar") == 2)
        df = pd.DataFrame({"foö": [13, 42, 42, 100], "bar": [1, 2, 3, 4], "z": 0.0})
        df_actual = cond.filter_df(df)
        df_expected = df.loc[(df["foö"] == 42) & (df["bar"] == 2)]
        pdt.assert_frame_equal(df_actual, df_expected)

    def test_filter_df_empty(self):
        cond = Conjunction([])
        df = pd.DataFrame({"foö": [13, 42, 42, 100], "bar": [1, 2, 3, 4], "z": 0.0})
        df_actual = cond.filter_df(df)
        pdt.assert_frame_equal(df_actual, df)

    def test_filter_df_nulls(self):
        cond = (C("foö") != 42.0) & (C("bar") != 2.0)
        df = pd.DataFrame(
            {"foö": [13, 42, np.nan, np.nan], "bar": [1, 2, 3, np.nan], "z": np.nan}
        )
        df_actual = cond.filter_df(df)
        df_expected = pd.DataFrame({"foö": [13.0], "bar": [1.0], "z": [np.nan]})
        pdt.assert_frame_equal(df_actual, df_expected)

    def test_hash(self):
        col = C("foö")
        cond1 = col < 10
        cond2 = col > 0
        cond3 = col != 10
        conj1a = cond1 & cond2
        conj1b = cond1 & cond2
        conj2 = cond1 & cond3
        assert hash(conj1a) == hash(conj1b)
        assert hash(conj1a) != hash(conj2)

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("sö == a", Conjunction([C("sö") == "a"])),
            ("sö == a & iö < 10", Conjunction([C("sö") == "a", C("iö") < 10])),
            ("(sö == a) & (iö < 10)", Conjunction([C("sö") == "a", C("iö") < 10])),
            ("", Conjunction([])),
            ("  ", Conjunction([])),
        ],
    )
    def test_from_string_ok(self, s, expected):
        all_types = {"sö": pa.string(), "bö": pa.bool_(), "iö": pa.int16()}
        actual = Conjunction.from_string(s, all_types)
        assert actual == expected

        s2 = str(actual)
        actual2 = Conjunction.from_string(s2, all_types)
        assert actual2 == actual

    @pytest.mark.parametrize(
        "obj,expected",
        [
            (
                # obj
                C("foö") > 1,
                # expected
                Conjunction([C("foö") > 1]),
            ),
            (
                # obj
                [C("foö") > 1],
                # expected
                Conjunction([C("foö") > 1]),
            ),
            (
                # obj
                [C("foö") > 1, C("bar") < 1],
                # expected
                Conjunction([C("foö") > 1, C("bar") < 1]),
            ),
            (
                # obj
                Conjunction([C("foö") > 1, C("bar") < 1]),
                # expected
                Conjunction([C("foö") > 1, C("bar") < 1]),
            ),
            (
                # obj
                None,
                # expected
                Conjunction([]),
            ),
        ],
    )
    def test_init_from_obj(self, obj, expected):
        actual = Conjunction(obj)
        assert actual == expected

    def test_fails(self):
        with pytest.raises(
            TypeError, match="Can only build conjunction out of conditions."
        ):
            Conjunction(1)

    def test_json_serialization_ok(self):
        conj = Conjunction(
            [
                EqualityCondition(column="foö", value=1.2),
                GreaterEqualCondition(column="foö", value=1.2),
                GreaterThanCondition(column="foö", value=1.2),
                InequalityCondition(column="foö", value=1.2),
                LessEqualCondition(column="foö", value=1.2),
                LessThanCondition(column="foö", value=1.2),
                InIntervalCondition(column="foö", start=1.2, stop=2.3),
                IsInCondition(column="foö", value=[1.2, 1.3]),
            ]
        )

        array_actual = conj.to_jsonarray()
        array_expected = [
            {"type": "EqualityCondition", "column": "foö", "value": 1.2},
            {"type": "GreaterEqualCondition", "column": "foö", "value": 1.2},
            {"type": "GreaterThanCondition", "column": "foö", "value": 1.2},
            {"type": "InequalityCondition", "column": "foö", "value": 1.2},
            {"type": "LessEqualCondition", "column": "foö", "value": 1.2},
            {"type": "LessThanCondition", "column": "foö", "value": 1.2},
            {"type": "InIntervalCondition", "column": "foö", "start": 1.2, "stop": 2.3},
            {"type": "IsInCondition", "column": "foö", "value": [1.2, 1.3]},
        ]
        assert array_actual == array_expected

        conj2 = Conjunction.from_jsonarray(array_actual)
        assert conj2 == conj

        # input not altered
        assert array_actual == array_expected

    @pytest.mark.parametrize(
        "array",
        [
            [{"type": "str"}],
            [{"type": "Condition"}],
            [{"type": "C"}],
            [{"type": "Conjunction"}],
            [{"type": "SimpleCondition"}],
            [{"type": "VirtualColumn"}],
            [{"type": "FooBar"}],
            [{"type": ""}],
            [{"type": " "}],
        ],
    )
    def test_json_serialization_fail_type(self, array):
        with pytest.raises(TypeError, match="Unknown condition class"):
            Conjunction.from_jsonarray(array)

    def test_json_serialization_fail_no_list(self):
        with pytest.raises(TypeError, match="jsonarray must be a list"):
            Conjunction.from_jsonarray({})

    def test_json_serialization_fail_no_cond_dict(self):
        with pytest.raises(TypeError, match="Condition in jsonarray must be a dict"):
            Conjunction.from_jsonarray([1])

    def test_json_serialization_fail_type_missing(self):
        with pytest.raises(ValueError, match="Missing type value for condition"):
            Conjunction.from_jsonarray([{}])
