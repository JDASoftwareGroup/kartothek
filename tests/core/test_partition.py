# -*- coding: utf-8 -*-

import pytest

from kartothek.core.partition import Partition


def test_roundtrip():
    expected = {"files": {"Queejeb3": "file.parquet"}}
    result = Partition.from_dict("partition_label", expected).to_dict()
    assert expected == result


def test_roundtrip_no_metadata():
    expected = {"files": {"Queejeb3": "file.parquet"}}
    result = Partition.from_dict("partition_label", expected).to_dict()
    assert expected == result


def test_roundtrip_empty_metadata():
    _input = {"files": {"Queejeb3": "file.parquet"}}
    expected = {"files": {"Queejeb3": "file.parquet"}}
    result = Partition.from_dict("partition_label", _input).to_dict()
    assert expected == result


def test_raise_on_erroneous_input():
    with pytest.raises(ValueError):
        Partition.from_dict(label="label", dct="some_not_supported_external_ref")


def test_eq():
    assert not (Partition("label") == Partition("other_label"))
    assert not (Partition("label") == Partition("label", files={"some": "file"}))

    assert Partition(label="label", files={"some": "file"}) == Partition(
        label="label", files={"some": "file"}
    )
