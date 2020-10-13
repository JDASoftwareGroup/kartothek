from storefact import get_store_from_url

from kartothek.serialization import ParquetSerializer
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
