import pytest

from kartothek.core.table import Table
from kartothek.io.testing.update import *  # noqa: F40


def _update_from_table(*args, **kwargs):
    partitions = []
    index_cols = None
    for part in args[0]:
        if part:
            partitions.append(dict(part["data"])["core"])
            index_cols = list(part.get("indices", {}).keys())

    table = (
        Table(
            dataset_uuid=kwargs["dataset_uuid"],
            store_factory=kwargs["store"],
            name="core",
        )
        .write()
        .add_partitions(partitions)
        .index_on(kwargs.get("secondary_indices", index_cols))
        .remove_partitions(kwargs.get("delete_scope"))
        .add_metadata(kwargs.get("metadata"))
        .sort_by(kwargs.get("sort_partitions_by"))
    )

    return table.commit()


@pytest.fixture()
def bound_update_dataset():
    return _update_from_table


def test_metadata_version():
    # Table interface expects other input
    pass


def test_raises_on_new_index_creation():
    # The table interface only supports the new-style input
    pass
