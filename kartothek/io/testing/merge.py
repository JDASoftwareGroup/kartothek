from collections import OrderedDict
from datetime import date

import pandas as pd
import pandas.testing as pdt

from kartothek.io_components.metapartition import SINGLE_TABLE

MERGE_TASKS = [
    {
        "left": SINGLE_TABLE,
        "right": "helper",
        "merge_kwargs": {"how": "left", "sort": False, "copy": False},
        "output_label": "first_output",
    },
    {
        "left": "first_output",
        "right": "PRED",
        "merge_kwargs": {"how": "left", "sort": False, "copy": False},
        "output_label": "final",
    },
]

MERGE_EXP_CL1 = pd.DataFrame(
    OrderedDict(
        [
            ("P", [1, 1]),
            ("L", [1, 1]),
            ("TARGET", [1, 1]),
            ("HORIZON", [1, 2]),
            ("info", ["a", "a"]),
            ("PRED", [10, 20]),
            ("DATE", pd.to_datetime([date(2010, 1, 1), date(2010, 1, 1)])),
        ]
    )
)

MERGE_EXP_CL2 = pd.DataFrame(
    OrderedDict(
        [
            ("P", [2, 2]),
            ("L", [2, 2]),
            ("TARGET", [2, 2]),
            ("HORIZON", [1, 2]),
            ("info", ["b", "b"]),
            ("PRED", [10, 20]),
            ("DATE", pd.to_datetime([date(2009, 12, 31), date(2009, 12, 31)])),
        ]
    )
)


def test_merge_datasets(
    dataset,
    evaluation_dataset,
    store_factory,
    store_session_factory,
    frozen_time,
    bound_merge_datasets,
):
    # In the __pipeline case, we also need to check that the write path is
    # correct, the tests for it are much larger.
    df_list = bound_merge_datasets(
        left_dataset_uuid=dataset.uuid,
        right_dataset_uuid=evaluation_dataset.uuid,
        store=store_session_factory,
        merge_tasks=MERGE_TASKS,
        match_how="prefix",
    )
    df_list = [mp.data for mp in df_list]

    # Two partitions
    assert len(df_list) == 2
    assert len(df_list[1]) == 1
    assert len(df_list[0]) == 1
    # By using values() this test is agnostic to the used key, which is
    # currently not of any importance
    pdt.assert_frame_equal(
        list(df_list[0].values())[0],
        MERGE_EXP_CL1,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
    pdt.assert_frame_equal(
        list(df_list[1].values())[0],
        MERGE_EXP_CL2,
        check_like=True,
        check_dtype=False,
        check_categorical=False,
    )
