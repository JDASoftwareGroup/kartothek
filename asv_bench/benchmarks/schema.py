# -*- coding: utf-8 -*-


from copy import deepcopy

from kartothek.core.common_metadata import (
    make_meta,
    validate_compatible,
    validate_shared_columns,
)
from kartothek.core.testing import get_dataframe_alltypes

from .config import AsvBenchmarkConfig


class TimeMakeMeta(AsvBenchmarkConfig):
    def setup(self):
        self.df = get_dataframe_alltypes()

    def time_make_meta(self):
        make_meta(self.df, origin="df")


class TimeValidateCompatible(AsvBenchmarkConfig):

    params = ([2, 10 ** 2, 10 ** 3, 10 ** 4], [True, False])
    timeout = 120.0

    param_names = ["num_schemas", "has_na"]

    def setup(self, num_schemas, has_na):
        self.df = get_dataframe_alltypes()
        schema = make_meta(self.df, origin="df")
        self.schemas = [deepcopy(schema) for _ in range(num_schemas)]
        if has_na:
            empty_schema = make_meta(self.df[0:0], origin="empty")
            # insert alternating empty schemas
            self.schemas[::2] = [
                deepcopy(empty_schema) for _ in range(len(self.schemas[::2]))
            ]

    def mem_schemas(self, num_schemas, has_na):
        return self.schemas

    def peakmem_validate_compatible(self, num_schemas, has_na):
        validate_compatible(self.schemas)

    def time_validate_compatible(self, num_schemas, has_na):
        validate_compatible(self.schemas)


class TimeValidateSharedColumns(AsvBenchmarkConfig):
    params = [2, 10 ** 2]
    timeout = 120.0

    param_names = ["num_schemas"]

    def setup(self, num_schemas):
        self.df = get_dataframe_alltypes()
        schema = make_meta(self.df, origin="df")
        self.schemas = [deepcopy(schema) for _ in range(num_schemas)]

    def peakmem_validate_shared_columns(self, num_schemas):
        validate_shared_columns(self.schemas)

    def time_validate_shared_columns(self, num_schemas):
        validate_shared_columns(self.schemas)
