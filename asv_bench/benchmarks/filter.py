from kartothek.serialization._generic import (
    filter_array_like,
    filter_df_from_predicates,
)
from kartothek.serialization.testing import get_dataframe_not_nested


class TimeFilterDF:

    params = ["conjunctions", "disjunctions"]

    def setup(self, predicate):
        if predicate == "conjunctions":
            self.predicate = [
                [
                    ("int16", ">", 123),
                    ("int32", "<", 321),
                    ("bool", "==", True),
                    ("bool", "==", True),
                ]
            ]
        elif predicate == "disjunctions":
            self.predicate = [
                [("int16", ">", 123)],
                [("int32", "<", 321)],
                [("int32", "<", 321)],
                [("int32", "<", 321)],
            ]
        self.df = get_dataframe_not_nested(10 ** 5)

    def time_filter_df_from_predicates(self, predicate):
        filter_df_from_predicates(self.df, self.predicate)


cols_to_filter = [
    # All numpy native types perform in the same order as integers therefore it
    # should be sufficient to test those (including datetime)
    "int64",
    # Object/strings are the true cost drivers and should be included in the benchmark
    "unicode",
]


class TimeFilterArray:
    params = cols_to_filter

    def setup(self, column):
        if column == "null":
            raise NotImplementedError()
        self.arr = (
            get_dataframe_not_nested(10 ** 5)
            .sample(frac=1.0)
            .reset_index(drop=True)[column]
            .values
        )
        self.value = self.arr[12345]

    def time_eq(self, column):
        filter_array_like(self.arr, "==", self.value)

    def time_ge(self, column):
        filter_array_like(self.arr, ">=", self.value)

    def time_le(self, column):
        filter_array_like(self.arr, "<=", self.value)


class TimeFilterArrayIn:
    params = (
        cols_to_filter,
        [10, 100, 1000],
        [10 ** 4, 10 ** 5, 10 ** 6],
    )
    param_names = ["column", "filter_size", "array_size", "enabled"]

    def setup(self, column, filter_size, array_size):

        if column == "null":
            raise NotImplementedError()
        ser = (
            get_dataframe_not_nested(array_size)
            .sample(frac=1.0)
            .reset_index(drop=True)[column]
        )
        self.arr = ser.values
        self.value = ser.sample(n=filter_size).unique()

    def time_in(self, column, filter_size, array_size):
        filter_array_like(self.arr, "in", self.value)


if __name__ == "__main__":
    restore = TimeFilterArrayIn()
    restore.setup("bool", 100, 100, -1)
    restore.time_in("bool", 100, 100, -1)
