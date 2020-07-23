import pandas as pd
import pyarrow as pa
import pytest

from kartothek.utils.converters import (
    converter_str,
    converter_str_set,
    converter_str_set_optional,
    converter_str_tupleset,
    converter_tuple,
    get_str_to_python_converter,
)


@pytest.mark.parametrize(
    "param,expected",
    [
        (
            # param
            "foo",
            # expected
            {"foo"},
        ),
        (
            # param
            "foo",
            # expected
            {"foo"},
        ),
        (
            # param
            b"foo",
            # expected
            {"foo"},
        ),
        (
            # param
            {"foo", "bar"},
            # expected
            {"foo", "bar"},
        ),
        (
            # param
            {"foo", b"foo"},
            # expected
            {"foo", "foo"},
        ),
        (
            # param
            ["foo", b"foo"],
            # expected
            {"foo", "foo"},
        ),
        (
            # param
            None,
            # expected
            set(),
        ),
    ],
)
def test_str_set(param, expected):
    actual = converter_str_set(param)
    assert isinstance(actual, frozenset)
    assert actual == expected
    assert all(isinstance(x, str) for x in actual)


@pytest.mark.parametrize(
    "param,expected",
    [
        (
            # param
            "foo",
            # expected
            {"foo"},
        ),
        (
            # param
            "foo",
            # expected
            {"foo"},
        ),
        (
            # param
            b"foo",
            # expected
            {"foo"},
        ),
        (
            # param
            {"foo", "bar"},
            # expected
            {"foo", "bar"},
        ),
        (
            # param
            {"foo", b"foo"},
            # expected
            {"foo", "foo"},
        ),
        (
            # param
            ["foo", b"foo"],
            # expected
            {"foo", "foo"},
        ),
        (
            # param
            None,
            # expected
            None,
        ),
    ],
)
def test_str_set_optional(param, expected):
    actual = converter_str_set_optional(param)
    assert actual == expected
    if actual is not None:
        assert isinstance(actual, frozenset)
        assert all(isinstance(x, str) for x in actual)


@pytest.mark.parametrize(
    "param,expected",
    [
        (
            # param
            "foo",
            # expected
            ("foo",),
        ),
        (
            # param
            "foo",
            # expected
            ("foo",),
        ),
        (
            # param
            b"foo",
            # expected
            ("foo",),
        ),
        (
            # param
            ["foo", "bar"],
            # expected
            ("foo", "bar"),
        ),
        (
            # param
            None,
            # expected
            (),
        ),
    ],
)
def test_str_tupleset_ok(param, expected):
    actual = converter_str_tupleset(param)
    assert isinstance(actual, tuple)
    assert actual == expected
    assert all(isinstance(x, str) for x in actual)


def test_str_tupleset_fail_duplicates():
    with pytest.raises(ValueError, match="Tuple-set contains duplicates: foo, foo"):
        converter_str_tupleset([b"foo", "foo"])


@pytest.mark.parametrize(
    "param",
    [
        # set
        {"foo", "bar"},
        # dict
        {"foo": 1, "bar": 2},
        # frozenset
        frozenset(("foo", "bar")),
    ],
)
def test_str_tupleset_fail_unstable(param):
    with pytest.raises(
        TypeError,
        match="which has type {} has an unstable iteration order".format(
            type(param).__name__
        ),
    ):
        converter_str_tupleset(param)


@pytest.mark.parametrize(
    "obj,expected",
    [
        (
            # obj
            "foo",
            # expected
            "foo",
        ),
        (
            # obj
            "foo",
            # expected
            "foo",
        ),
        (
            # obj
            b"foo",
            # expected
            "foo",
        ),
    ],
)
def test_str_ok(obj, expected):
    actual = converter_str(obj)
    assert actual == expected


@pytest.mark.parametrize(
    "obj,msg",
    [
        (
            # obj
            1,
            # msg
            "Object of type int is not a string: 1",
        ),
        (
            # obj
            ["a", "b"],
            # msg
            "Object of type list is not a string: {}".format(["a", "b"]),
        ),
        (
            # obj
            ["a"],
            # msg
            "Object of type list is not a string: {}".format(["a"]),
        ),
    ],
)
def test_str_fail(obj, msg):
    with pytest.raises(TypeError) as exc:
        converter_str(obj)
    assert str(exc.value) == msg


def test_str_rejects_none():
    with pytest.raises(TypeError) as exc:
        converter_str(None)
    assert str(exc.value) == "Object of type NoneType is not a string: None"


@pytest.mark.parametrize(
    "param,expected",
    [
        (
            # param
            "foo",
            # expected
            ("foo",),
        ),
        (
            # param
            "foo",
            # expected
            ("foo",),
        ),
        (
            # param
            b"foo",
            # expected
            (b"foo",),
        ),
        (
            # param
            ["foo", "bar"],
            # expected
            ("foo", "bar"),
        ),
        (
            # param
            ["foo", None],
            # expected
            ("foo", None),
        ),
        (
            # param
            None,
            # expected
            (),
        ),
    ],
)
def test_tuple(param, expected):
    actual = converter_tuple(param)
    assert actual == expected


@pytest.mark.parametrize(
    "s,pa_type,expected",
    [
        (
            # s
            "1",
            # pa_type
            pa.int16(),
            # expected
            1,
        ),
        (
            # s
            "1",
            # pa_type
            pa.int64(),
            # expected
            1,
        ),
        (
            # s
            "1",
            # pa_type
            pa.float32(),
            # expected
            1.0,
        ),
        (
            # s
            "1.2",
            # pa_type
            pa.float32(),
            # expected
            1.2,
        ),
        (
            # s
            "2018",
            # pa_type
            pa.timestamp("ns"),
            # expected
            pd.Timestamp("2018"),
        ),
        (
            # s
            "true",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "TRUE",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "false",
            # pa_type
            pa.bool_(),
            # expected
            False,
        ),
        (
            # s
            "t",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "f",
            # pa_type
            pa.bool_(),
            # expected
            False,
        ),
        (
            # s
            "1",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "0",
            # pa_type
            pa.bool_(),
            # expected
            False,
        ),
        (
            # s
            "yes",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "no",
            # pa_type
            pa.bool_(),
            # expected
            False,
        ),
        (
            # s
            "y",
            # pa_type
            pa.bool_(),
            # expected
            True,
        ),
        (
            # s
            "n",
            # pa_type
            pa.bool_(),
            # expected
            False,
        ),
        (
            # s
            " foo ",
            # pa_type
            pa.string(),
            # expected
            " foo ",
        ),
        (
            # s
            '"foo"',
            # pa_type
            pa.string(),
            # expected
            "foo",
        ),
        (
            # s
            "'foo'",
            # pa_type
            pa.string(),
            # expected
            "foo",
        ),
        (
            # s
            '"',
            # pa_type
            pa.string(),
            # expected
            '"',
        ),
        (
            # s
            '"foo',
            # pa_type
            pa.string(),
            # expected
            '"foo',
        ),
    ],
)
def test_get_str_to_python_converter_ok(s, pa_type, expected):
    converter = get_str_to_python_converter(pa_type)
    actual = converter(s)
    assert actual == expected
    assert isinstance(actual, type(expected))


def test_get_str_to_python_converter_unknown_type():
    pa_type = pa.struct([("f1", pa.int32()), ("f2", pa.bool_())])
    with pytest.raises(ValueError) as exc:
        get_str_to_python_converter(pa_type)
    assert str(exc.value) == "Cannot handle type struct<f1: int32, f2: bool>"


@pytest.mark.parametrize(
    "s,pa_type",
    [
        (
            # s
            "foo",
            # pa_type
            pa.bool_(),
        ),
        (
            # s
            "",
            # pa_type
            pa.bool_(),
        ),
        (
            # s
            "1.2",
            # pa_type
            pa.int32(),
        ),
    ],
)
def test_get_str_to_python_converter_failes(s, pa_type):
    converter = get_str_to_python_converter(pa_type)
    with pytest.raises(ValueError):
        converter(s)
