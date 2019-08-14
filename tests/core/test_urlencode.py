# -*- coding: utf-8 -*-

from urllib.parse import quote as quote_python

from kartothek.core.urlencode import quote as quote_ktk
from kartothek.core.urlencode import unquote as unquote_ktk

TEST_STRING = "Test string with lots of special characters !@#$%^&*+=()[]\\{}<>?|'\"-_~`☺✌☕file.par"


def test_urlquoting_backwards_compatibility():
    """
    This tests asserts that unquoting strings encoded with urlquote does produce the same result as
    unquoting strings quoted with pythons urllib (the encodings differ for Python < 3.7).
    """
    assert unquote_ktk(quote_ktk(TEST_STRING)) == unquote_ktk(quote_python(TEST_STRING))
