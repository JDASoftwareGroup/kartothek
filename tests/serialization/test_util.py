# -*- coding: utf-8 -*-
import pytest

from kartothek.serialization._util import ensure_unicode_string_type


@pytest.mark.parametrize(
    "obj,expected", [("tüst", "tüst"), ("tüst".encode("utf8"), "tüst")]
)
def test_ensure_unicode_string_types(obj, expected):
    actual = ensure_unicode_string_type(obj)
    assert type(actual) == str
    assert actual == expected
