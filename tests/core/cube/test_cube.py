import pytest

from kartothek.core.cube.cube import Cube


def test_defaults():
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    assert cube.seed_dataset == "seed"
    assert isinstance(cube.seed_dataset, str)

    assert cube.index_columns == set()


def test_converters():
    cube = Cube(
        dimension_columns=b"my_dim",
        partition_columns=b"my_part",
        uuid_prefix=b"my_prefix",
        seed_dataset=b"my_seed",
        index_columns=b"my_index",
    )

    assert cube.dimension_columns == ("my_dim",)
    assert all(isinstance(s, str) for s in cube.dimension_columns)

    assert cube.partition_columns == ("my_part",)
    assert all(isinstance(s, str) for s in cube.partition_columns)

    assert cube.uuid_prefix == "my_prefix"
    assert isinstance(cube.uuid_prefix, str)

    assert cube.seed_dataset == "my_seed"
    assert isinstance(cube.seed_dataset, str)

    assert cube.index_columns == {"my_index"}
    assert all(isinstance(s, str) for s in cube.index_columns)


def test_frozen():
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    with pytest.raises(AttributeError):
        cube.uuid_prefix = "cube2"


def test_ktk_dataset_uuid():
    cube = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")

    assert cube.ktk_dataset_uuid(b"foo") == "cube++foo"
    assert isinstance(cube.ktk_dataset_uuid(b"foo"), str)

    with pytest.raises(ValueError) as exc:
        cube.ktk_dataset_uuid("f++")
    assert (
        str(exc.value)
        == 'ktk_cube_dataset_id ("f++") must not contain UUID separator ++'
    )

    with pytest.raises(ValueError) as exc:
        cube.ktk_dataset_uuid("f ")
    assert (
        str(exc.value) == 'ktk_cube_dataset_id ("f ") is not compatible with kartothek'
    )


def test_ktk_index_columns():
    cube = Cube(
        dimension_columns=["dim1", "dim2"],
        partition_columns=["part1", "part2"],
        index_columns=["index1", "index2"],
        uuid_prefix="cube",
    )
    assert cube.ktk_index_columns == {
        "dim1",
        "dim2",
        "part1",
        "part2",
        "index1",
        "index2",
    }
    assert all(isinstance(s, str) for s in cube.ktk_index_columns)


def test_init_fail_partition_columns_subsetof_index_columns():
    with pytest.raises(ValueError) as exc:
        Cube(
            dimension_columns=["x", "y", "z"],
            partition_columns=["p", "y", "z"],
            uuid_prefix="cube",
        )
    assert (
        str(exc.value)
        == "partition_columns cannot share columns with dimension_columns, but share the following: y, z"
    )


def test_init_fail_index_columns_subsetof_dimension_columns():
    with pytest.raises(ValueError) as exc:
        Cube(
            dimension_columns=["x", "y", "z"],
            partition_columns=["p"],
            uuid_prefix="cube",
            index_columns=["i", "y", "z"],
        )
    assert (
        str(exc.value)
        == "index_columns cannot share columns with dimension_columns, but share the following: y, z"
    )


def test_init_fail_index_columns_subsetof_partition_columns():
    with pytest.raises(ValueError) as exc:
        Cube(
            dimension_columns=["x", "y", "z"],
            partition_columns=["p", "q", "r"],
            uuid_prefix="cube",
            index_columns=["i", "q", "r"],
        )
    assert (
        str(exc.value)
        == "index_columns cannot share columns with partition_columns, but share the following: q, r"
    )


def test_init_fail_illegal_uuid_prefix_sep():
    with pytest.raises(ValueError) as exc:
        Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cu++be")
    assert str(exc.value) == 'uuid_prefix ("cu++be") must not contain UUID separator ++'


def test_init_fail_illegal_uuid_prefix_ktk():
    with pytest.raises(ValueError) as exc:
        Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cu be")
    assert str(exc.value) == 'uuid_prefix ("cu be") is not compatible with kartothek'


def test_init_fail_illegal_seed_dataset_sep():
    with pytest.raises(ValueError) as exc:
        Cube(
            dimension_columns=["x"],
            partition_columns=["p"],
            uuid_prefix="cube",
            seed_dataset="se++ed",
        )
    assert (
        str(exc.value) == 'seed_dataset ("se++ed") must not contain UUID separator ++'
    )


def test_init_fail_illegal_seed_dataset_ktk():
    with pytest.raises(ValueError) as exc:
        Cube(
            dimension_columns=["x"],
            partition_columns=["p"],
            uuid_prefix="cube",
            seed_dataset="se ed",
        )
    assert str(exc.value) == 'seed_dataset ("se ed") is not compatible with kartothek'


def test_init_fail_empty_dimension_columns():
    with pytest.raises(ValueError) as exc:
        Cube(dimension_columns=[], partition_columns=["p"], uuid_prefix="cube")
    assert str(exc.value) == "dimension_columns must not be empty"


def test_init_fail_empty_partition_columns():
    with pytest.raises(ValueError) as exc:
        Cube(dimension_columns=["x"], partition_columns=[], uuid_prefix="cube")
    assert str(exc.value) == "partition_columns must not be empty"


def test_copy_simple():
    cube1 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube1")
    cube2 = cube1.copy(uuid_prefix="cube2")
    assert cube1.uuid_prefix == "cube1"
    assert cube2.uuid_prefix == "cube2"


def test_copy_converts():
    cube1 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube1")
    cube2 = cube1.copy(dimension_columns="foo")
    assert cube2.dimension_columns == ("foo",)


def test_copy_validates():
    cube1 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube1")
    with pytest.raises(ValueError) as exc:
        cube1.copy(uuid_prefix="cube2++")
    assert (
        str(exc.value) == 'uuid_prefix ("cube2++") must not contain UUID separator ++'
    )


def test_hash():
    cube1 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    cube2 = Cube(dimension_columns=["x"], partition_columns=["p"], uuid_prefix="cube")
    cube3 = Cube(
        dimension_columns=["x", "y"], partition_columns=["p"], uuid_prefix="cube"
    )

    assert hash(cube1) == hash(cube2)
    assert hash(cube3) != hash(cube1)
