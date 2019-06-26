# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from storefact import get_store_from_url

from kartothek.serialization import ParquetSerializer
from kartothek.serialization._generic import filter_df_from_predicates
from kartothek.serialization.testing import get_dataframe_not_nested


class TimeRestore:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    params = [(10 ** 3, 10 ** 4), (10, 10 ** 2, 10 ** 3)]
    param_names = ["num_rows", "chunk_size"]

    def setup(self, num_rows, chunk_size):
        self.df = get_dataframe_not_nested(num_rows)
        self.serialiser = ParquetSerializer(chunk_size=chunk_size)
        self.store = get_store_from_url("memory://")
        self.key = self.serialiser.store(self.store, "key_prefix", self.df)
        self.predicates = [[("int16", "==", 123)]]

    def time_predicate_pushdown(self, num_rows, chunk_size):
        self.serialiser.restore_dataframe(
            self.store,
            self.key,
            predicate_pushdown_to_io=True,
            predicates=self.predicates,
        )


class TimeFilterDF:

    params = ["conjunctions", "disjunctions"]

    def setup(self, predicate):
        if "conjunctions":
            self.predicate = [
                [
                    ("int16", ">", 123),
                    ("int32", "<", 321),
                    ("bool", "==", True),
                    ("bool", "==", True),
                ]
            ]
        elif "disjunctions":
            self.predicate = [
                [("int16", ">", 123)],
                [("int32", "<", 321)],
                [("int32", "<", 321)],
                [("int32", "<", 321)],
            ]
        self.df = get_dataframe_not_nested(10 ** 5)

    def time_filter_df_from_predicates(self, predicate):
        filter_df_from_predicates(self.df, self.predicate)


if __name__ == "__main__":
    restore = TimeRestore()
    restore.setup(10 ** 5, 100)
    restore.time_predicate_pushdown(10 ** 5, 100)
