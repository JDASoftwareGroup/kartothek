def test_fail_skv_missing(cli):
    result = cli("--store=cubes", "my_cube", "info")
    assert result.exit_code == 2
    assert "Error: Could not open load store YAML:" in result.output


def test_fail_wrong_store(cli, skv):
    result = cli("--store=cubi", "my_cube", "info")
    assert result.exit_code == 2
    assert "Error: Could not find store cubi in skv.yml" in result.output


def test_fail_cube_missing(cli, skv):
    result = cli("--store=cubes", "my_cube", "info")
    assert result.exit_code == 2
    assert "Error: Could not load cube:" in result.output


def test_simple(cli, built_cube, skv):
    result = cli("--store=cubes", "my_cube", "info")
    assert result.exit_code == 0
    assert (
        result.output
        == """Infos
UUID Prefix:        my_cube
Dimension Columns:
  - x: int64
  - y: int64
Partition Columns:
  - p: int64
  - q: string
Index Columns:
  - i1: bool
  - i2: timestamp[ns]
Seed Dataset:      source

Dataset: enrich
Partition Keys:
  - part: int64
  - q: string
Partitions: 2
Metadata:
  {
    "creation_time": "2019-02-28T13:01:17",
    "ktk_cube_dimension_columns": [
      "x",
      "y"
    ],
    "ktk_cube_is_seed": false,
    "ktk_cube_partition_columns": [
      "p",
      "q"
    ]
  }
Dimension Columns:
  - y: int64
Payload Columns:
  - i2: timestamp[ns]
  - part: int64
  - v2: list<item: int64>

Dataset: source
Partition Keys:
  - p: int64
  - q: string
Partitions: 2
Metadata:
  {
    "creation_time": "2018-01-31T14:03:22",
    "ktk_cube_dimension_columns": [
      "x",
      "y"
    ],
    "ktk_cube_is_seed": true,
    "ktk_cube_partition_columns": [
      "p",
      "q"
    ]
  }
Dimension Columns:
  - x: int64
  - y: int64
Payload Columns:
  - i1: bool
  - v1: double
"""
    )
