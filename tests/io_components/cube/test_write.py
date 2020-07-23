import pytest

from kartothek.core.cube.constants import (
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
)
from kartothek.core.cube.cube import Cube
from kartothek.io_components.cube.write import (
    check_provided_metadata_dict,
    prepare_ktk_metadata,
)


@pytest.fixture
def cube():
    return Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube__uuid",
        seed_dataset="source",
    )


def test_prepare_ktk_metadata_simple(cube):
    metadata = prepare_ktk_metadata(cube, "source", None)
    assert metadata == {
        KTK_CUBE_METADATA_DIMENSION_COLUMNS: ["x"],
        KTK_CUBE_METADATA_PARTITION_COLUMNS: ["p"],
        KTK_CUBE_METADATA_KEY_IS_SEED: True,
    }


def test_prepare_ktk_metadata_no_source(cube):
    metadata = prepare_ktk_metadata(cube, "no_source", None)
    assert metadata == {
        KTK_CUBE_METADATA_DIMENSION_COLUMNS: ["x"],
        KTK_CUBE_METADATA_PARTITION_COLUMNS: ["p"],
        KTK_CUBE_METADATA_KEY_IS_SEED: False,
    }


def test_prepare_ktk_metadata_usermeta(cube):
    metadata = prepare_ktk_metadata(
        cube,
        "no_source",
        {"source": {"bla": "blub"}, "no_source": {"user_key0": "value0"}},
    )
    assert metadata == {
        KTK_CUBE_METADATA_DIMENSION_COLUMNS: ["x"],
        KTK_CUBE_METADATA_PARTITION_COLUMNS: ["p"],
        KTK_CUBE_METADATA_KEY_IS_SEED: False,
        "user_key0": "value0",
    }


def test_check_provided_metadata_dict_wrong_type():
    with pytest.raises(
        TypeError, match="Provided metadata should be a dict but is list"
    ):
        check_provided_metadata_dict([], [])


def test_check_provided_metadata_dict_wrong_type_nested():
    with pytest.raises(
        TypeError, match="Provided metadata for dataset a should be a dict but is list"
    ):
        check_provided_metadata_dict({"a": []}, ["a"])


def test_check_provided_metadata_dict_unknown_ids():
    with pytest.raises(
        ValueError,
        match="Provided metadata for otherwise unspecified ktk_cube_dataset_ids: a, b",
    ):
        check_provided_metadata_dict({"a": {}, "b": {}, "c": {}}, ["c"])
