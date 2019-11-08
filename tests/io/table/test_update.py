import pytest

from kartothek.core.table import Table
from kartothek.io.testing.update import *  # noqa: F40


def _update_from_table(*args, **kwargs):
    partitions = []

    for part in args[0]:
        if part:
            partitions.append(dict(part["data"])["core"])
    table = Table(
        dataset_uuid=kwargs["dataset_uuid"], store_factory=kwargs["store"], name="core"
    )
    if "dataset_uuid" in kwargs:
        del kwargs["dataset_uuid"]
    del kwargs["store"]
    if "label_filter" in kwargs:
        del kwargs["label_filter"]
    if "factory" in kwargs:
        del kwargs["factory"]

    table = table.write(**kwargs).add_partitions(partitions)

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
