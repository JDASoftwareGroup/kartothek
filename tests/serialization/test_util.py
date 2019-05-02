# -*- coding: utf-8 -*-
import pytest
import six

from kartothek.serialization._util import ensure_unicode_string_type


@pytest.mark.parametrize(
    "obj,expected", [(u"tüst", u"tüst"), (u"tüst".encode("utf8"), u"tüst")]
)
def test_ensure_unicode_string_types(obj, expected):
    actual = ensure_unicode_string_type(obj)
    assert type(actual) == six.text_type
    assert actual == expected
