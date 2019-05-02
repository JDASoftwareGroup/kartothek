import pytest

from kartothek.serialization import CsvSerializer, ParquetSerializer


@pytest.mark.parametrize(
    "serialiser,expected",
    [
        (CsvSerializer(), "CsvSerializer(compress=True)"),
        (CsvSerializer(compress=False), "CsvSerializer(compress=False)"),
        (
            ParquetSerializer(),
            "ParquetSerializer(compression='SNAPPY', chunk_size=None)",
        ),
        (
            ParquetSerializer(compression="GZIP"),
            "ParquetSerializer(compression='GZIP', chunk_size=None)",
        ),
        (
            ParquetSerializer(chunk_size=1000),
            "ParquetSerializer(compression='SNAPPY', chunk_size=1000)",
        ),
    ],
)
def test_repr(serialiser, expected):
    actual = repr(serialiser)
    assert actual == expected


@pytest.mark.parametrize(
    "obj1,obj2,expected",
    [
        (CsvSerializer(), CsvSerializer(), True),
        (CsvSerializer(), CsvSerializer(compress=False), False),
        (CsvSerializer(), ParquetSerializer(), False),
        (ParquetSerializer(), ParquetSerializer(), True),
        (ParquetSerializer(), ParquetSerializer(compression="GZIP"), False),
        (ParquetSerializer(), ParquetSerializer(chunk_size=1000), False),
        (ParquetSerializer(), CsvSerializer(), False),
    ],
)
def test_eq(obj1, obj2, expected):
    actual = obj1 == obj2
    assert actual == expected

    not_actual = obj1 != obj2
    not_expected = not expected
    assert not_actual == not_expected
