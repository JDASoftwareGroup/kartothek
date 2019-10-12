from .utils import create_dataset


def test_garbage_collect_idempotent(store_factory, garbage_collect_callable):
    """Check that garbage collection does nothing when there is no garbage."""

    create_dataset("uuid", store_factory, 4)

    keys_before = set(store_factory().keys())
    garbage_collect_callable("uuid", store_factory)
    keys_after = set(store_factory().keys())
    assert keys_before == keys_after


def _test_gc(uuid, store_factory, garbage_collect_callable):
    store = store_factory()

    keys_before = set(store.keys())

    # Add a non-tracked table file
    store.put("{}/table/trash.parquet".format(uuid), b"trash")

    # Add a non-tracked index file
    store.put("{}/indices/trash.parquet".format(uuid), b"trash")

    garbage_collect_callable(uuid, store_factory)

    keys_after = set(store.keys())
    assert keys_before == keys_after


def test_gc_tables(store_factory, garbage_collect_callable):
    create_dataset("uuid", store_factory, 4)
    _test_gc("uuid", store_factory, garbage_collect_callable)


def test_gc_without_secondary_indices(
    store_factory, garbage_collect_callable, dataset_function
):
    _test_gc("dataset_uuid", store_factory, garbage_collect_callable)
