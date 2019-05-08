# -*- coding: utf-8 -*-

# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import datetime

import numpy as np
import pandas as pd

from kartothek.core.common_metadata import make_meta
from kartothek.io_components.metapartition import MetaPartition

from .config import AsvBenchmarkConfig


class TimeMetaPartition(AsvBenchmarkConfig):
    params = (
        [10 ** 5, 10 ** 6],
        [
            (np.int64, 123456789),
            (str, "abcdefgh"),
            ("object", datetime.date(2018, 1, 1)),
        ],
    )

    def setup(self, num_rows, dtype):
        df = pd.DataFrame(
            {"primary_key": pd.Series(dtype[1], index=range(num_rows), dtype=dtype[0])}
        )
        self.schema = make_meta(df, partition_keys=["primary_key"], origin="df")
        self.df = df.drop("primary_key", axis=1)
        self.mp = MetaPartition(
            label="primary_key={}/base_label".format(dtype[0]),
            metadata_version=4,
            table_meta={"table": self.schema},
        )

    def time_reconstruct_index(self, num_rows, dtype):

        self.mp._reconstruct_index_columns(
            df=self.df,
            key_indices=[("primary_key", str(dtype[1]))],
            table="table",
            columns=None,
            categories=None,
            date_as_object=False,
        )

    def time_reconstruct_index_categorical(self, num_rows, dtype):
        self.mp._reconstruct_index_columns(
            df=self.df,
            key_indices=[("primary_key", str(dtype[1]))],
            table="table",
            columns=None,
            categories={"table": ["primary_key"]},
            date_as_object=False,
        )
