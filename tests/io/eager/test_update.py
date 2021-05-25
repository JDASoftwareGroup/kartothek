import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from kartothek.io.eager import (
    commit_dataset,
    create_empty_dataset_header,
    update_dataset_from_dataframes,
    write_single_partition,
)
from kartothek.io.testing.update import *  # noqa: F40


@pytest.fixture()
def bound_update_dataset():
    return update_dataset_from_dataframes


def test_create_empty_header_from_pyarrow_schema(store_factory):
    # GH228
    df = pd.DataFrame(
        [{"part": 1, "id": 1, "col1": "abc"}, {"part": 2, "id": 2, "col1": np.nan}]
    )
    dataset_uuid = "sample_ds"
    schema = pa.Schema.from_pandas(df)

    dm = create_empty_dataset_header(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        table_meta={"table": schema},
        partition_on=["part"],
    )

    new_partitions = [
        write_single_partition(
            store=store_factory,
            dataset_uuid=dataset_uuid,
            data=[{"table": df.loc[df["part"] == 1]}],
            partition_on=["part"],
        )
    ]
    assert len(dm.partitions) == 0
    dm = commit_dataset(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        new_partitions=new_partitions,
        partition_on=["part"],
    )

    assert len(dm.partitions) == 1
