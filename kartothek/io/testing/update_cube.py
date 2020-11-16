from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from kartothek.core.cube.conditions import C
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadata
from kartothek.io.eager import read_table
from kartothek.io.eager_cube import build_cube, extend_cube, query_cube


def _write_cube(function_store) -> Tuple[pd.DataFrame, Cube]:
    """
    Write a cube with dimension column "x" and partition column "p"

    returns the 'source' and 'enrich' dataframes and the cube specification.
    """
    df_source = pd.DataFrame(
        {
            "i1": [10, 11, 12, 13],
            "p": [0, 0, 1, 1],
            "v1": [10, 11, 12, 13],
            "x": [0, 1, 2, 3],
        }
    )
    cube = Cube(
        dimension_columns=["x"],
        partition_columns=["p"],
        uuid_prefix="cube",
        seed_dataset="source",
        index_columns=["i1", "i2", "i3"],
    )
    build_cube(
        data={"source": df_source},
        cube=cube,
        store=function_store,
        metadata={"source": {"meta_at_create": "data"}},
    )
    return df_source, cube


def _extend_cube(cube, function_store) -> pd.DataFrame:
    # extend the existing cube by a dataset 'ex' with columns a = x + 1000
    df = pd.DataFrame({"a": [1000, 1001], "p": [0, 1], "x": [0, 2]})
    extend_cube({"ex": df}, cube, function_store)
    return df


@pytest.mark.parametrize(
    "remove_partitions,new_partitions",
    [
        # only append:
        ([], [4, 5]),
        # do nothing:
        ([], []),
        # partial overwrite with new data for p=0
        ([0], [0, 1, 4]),
        # explicitly remove p=0 without overwriting it
        ([0], [1, 4]),
        # overwrite all:
        ([0, 1], [0, 1]),
    ],
)
def test_update_partitions(driver, function_store, remove_partitions, new_partitions):
    df_source, cube = _write_cube(function_store)

    df_source_new = pd.DataFrame(
        {
            "i1": range(200, 200 + len(new_partitions)),
            "p": np.array(new_partitions, np.int64),
            "v1": range(300, 300 + len(new_partitions)),
            "x": range(100, 100 + len(new_partitions)),
        }
    )

    # what should remain of the old data:
    df_source_of_old = df_source.loc[~df_source["p"].isin(set(remove_partitions))]
    df_source_expected_after = pd.concat(
        [df_source_of_old, df_source_new], sort=False, ignore_index=True
    )

    remove_conditions = C("p").isin(remove_partitions)

    result = driver(
        data={"source": df_source_new},
        remove_conditions=remove_conditions,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids={"source"},
        metadata={"source": {"some_new_meta": 42}},
    )

    assert set(result.keys()) == {"source"}

    dm_source_after = DatasetMetadata.load_from_store(
        cube.ktk_dataset_uuid("source"), function_store(), load_all_indices=True
    )

    assert "some_new_meta" in dm_source_after.metadata
    assert "meta_at_create" in dm_source_after.metadata

    # check values for "p" are as expected:
    expected_p_source = (set(df_source["p"].unique()) - set(remove_partitions)) | set(
        new_partitions
    )
    assert set(dm_source_after.indices["p"].index_dct) == expected_p_source

    df_read = query_cube(cube, function_store)[0]

    assert set(df_read.columns) == set(df_source_expected_after.columns)

    for df in (df_read, df_source_expected_after):
        df.sort_values("x", inplace=True)
        df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(df_read, df_source_expected_after)


@pytest.mark.parametrize(
    "ktk_cube_dataset_ids", [{"source", "ex"}, {"source"}, {"ex"}, set()]
)
def test_update_respects_ktk_cube_dataset_ids(
    driver, function_store, ktk_cube_dataset_ids
):
    df_source, cube = _write_cube(function_store)
    df_ex = _extend_cube(cube, function_store)

    remove_conditions = C("p") == 0

    # This implicitly also tests that `data={}` behaves as expected and still deletes partitions
    # as requested via ktk_cube_dataset_ids and remove_conditions
    result = driver(
        data={},
        remove_conditions=remove_conditions,
        cube=cube,
        store=function_store,
        ktk_cube_dataset_ids=ktk_cube_dataset_ids,
    )
    assert set(result) == ktk_cube_dataset_ids
    df_read = query_cube(cube, function_store)[0]

    # expected result: df_source left joined with df_ex; choosing the subset of p!=0 from each
    # that is in `ktk_cube_dataset_ids`:
    if "source" in ktk_cube_dataset_ids:
        df_source = df_source.loc[df_source["p"] != 0]
    if "ex" in ktk_cube_dataset_ids:
        df_ex = df_ex.loc[df_ex["p"] != 0]
    df_expected = df_source.merge(df_ex[["x", "a"]], how="left", on="x")
    df_expected = df_expected[sorted(df_expected.columns)]
    pd.testing.assert_frame_equal(df_read, df_expected)

    # test "ex" separately, because the test above based on the *left* merge does not tell us much about
    # "ex" in case the partitions were removed from "source"
    df_ex_read = read_table(cube.ktk_dataset_uuid("ex"), function_store)
    if "ex" in ktk_cube_dataset_ids:
        assert set(df_ex_read["p"]) == {1}
    else:
        assert set(df_ex_read["p"]) == {0, 1}
