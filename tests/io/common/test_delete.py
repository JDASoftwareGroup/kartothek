from .utils import create_dataset


def test_delete_dataset(store_factory, metadata_version, bound_delete_dataset):
    """
    Ensure that a dataset can be deleted
    """
    create_dataset("dataset", store_factory, metadata_version)

    store = store_factory()
    assert len(list(store.keys())) > 0
    bound_delete_dataset("dataset", store_factory)
    assert len(list(store.keys())) == 0


def test_delete_single_dataset(store_factory, metadata_version, bound_delete_dataset):
    """
    Ensure that only the specified dataset is deleted
    """
    create_dataset("dataset", store_factory, metadata_version)
    create_dataset("another_dataset", store_factory, metadata_version)
    store = store_factory()
    amount_of_keys = len(list(store.keys()))
    assert len(list(store.keys())) > 0
    bound_delete_dataset("dataset", store_factory)
    assert len(list(store.keys())) == amount_of_keys / 2, store.keys()


def test_delete_only_dataset(store_factory, metadata_version, bound_delete_dataset):
    """
    Ensure that files including the UUID but not starting with it
    are not deleted
    """
    create_dataset("UUID", store_factory, metadata_version)

    store = store_factory()
    store.put(key="prefixUUID", data=b"")
    bound_delete_dataset("UUID", store_factory)
    assert "prefixUUID" in store.keys()


def test_delete_missing_dataset(store_factory, store_factory2, bound_delete_dataset):
    """
    Ensure that a dataset can be deleted even though some keys are already removed.
    """
    metadata_version = 4
    create_dataset("dataset", store_factory, metadata_version)

    store = store_factory()
    keys = sorted(store.keys())
    assert len(keys) > 0

    store2 = store_factory2()

    for missing in keys:
        if missing == "dataset.by-dataset-metadata.json":
            continue

        for k in keys:
            if k != missing:
                store2.put(k, store.get(k))

        bound_delete_dataset("dataset", store_factory2)
        assert len(list(store2.keys())) == 0


def test_delete_dataset_unreferenced_files(
    store_factory, metadata_version, bound_delete_dataset
):
    """
    Ensure that unreferenced files of a dataset are also removed when a dataset is deleted
    """
    uuid = "dataset"
    create_dataset(uuid, store_factory, metadata_version)

    store = store_factory()
    store.put(f"{uuid}/table/trash.parquet", b"trash")

    assert len(list(store.keys())) > 0
    bound_delete_dataset(uuid, store_factory)

    assert len(list(store.keys())) == 0
