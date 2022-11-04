import pytest

from kartothek.core.factory import DatasetFactory
from kartothek.core.utils import lazy_store
from kartothek.io.eager import copy_dataset, read_table, store_dataframes_as_dataset
from kartothek.serialization.testing import get_dataframe_not_nested
from kartothek.utils.ktk_adapters import get_dataset_keys

SRC_DS_UUID = "test_copy_ds_with_index"
TGT_DS_UUID = "copy_target"


@pytest.fixture
def dataset_to_copy(store):
    df = get_dataframe_not_nested(10)
    store_dataframes_as_dataset(
        dfs=[df],
        dataset_uuid=SRC_DS_UUID,
        store=store,
        partition_on=[df.columns[0]],
        secondary_indices=[df.columns[1]],
    )


def assert_target_ktk_readable(tgt_store, tgt_ds):
    """
    Try to read the target dataset using high level KTK functionality
    """
    df_result = read_table(
        store=tgt_store,
        dataset_uuid=tgt_ds,
    )
    assert df_result is not None
    assert len(df_result) == 10
    df_result = read_table(
        store=tgt_store, dataset_uuid=tgt_ds, predicates=[[("bool", "==", True)]]
    )
    assert len(df_result) == 5
    df_result = read_table(
        store=tgt_store, dataset_uuid=tgt_ds, predicates=[[("bytes", "==", b"2")]]
    )
    assert len(df_result) == 1


def assert_target_keys(src_store, src_uuid, tgt_store, tgt_uuid):
    """
    Check that the expected keys exist in the target data set, and the corresponding
    values are equal to the source data set (or modified as expected)
    """
    df_source = DatasetFactory(
        dataset_uuid=src_uuid,
        store_factory=lazy_store(src_store),
    )
    src_keys = get_dataset_keys(df_source.dataset_metadata)
    df_target = DatasetFactory(
        dataset_uuid=tgt_uuid,
        store_factory=lazy_store(tgt_store),
    )
    tgt_keys = get_dataset_keys(df_target.dataset_metadata)

    for src_key in src_keys:
        # check for each source key if the corresponding target key exists
        tgt_key = src_key.replace(src_uuid, tgt_uuid)
        assert tgt_key in tgt_keys

        # check if the files for source and target key are equal (exception:
        # metadata => here the target must contain the modified metadata)
        b1 = src_store.get(src_key)
        b2 = tgt_store.get(tgt_key)

        if tgt_key.endswith("by-dataset-metadata.json"):
            b1_mod = b1.decode("utf-8").replace(src_uuid, tgt_uuid).encode("utf-8")
            assert b1_mod == b2
        else:
            assert b1 == b2


def test_copy_rename_eager_same_store(dataset_to_copy, store):
    """
    Copies and renames DS within one store
    """
    copy_dataset(
        source_dataset_uuid=SRC_DS_UUID,
        target_dataset_uuid=TGT_DS_UUID,
        store=store,
        target_store=store,
    )
    assert_target_keys(store, SRC_DS_UUID, store, TGT_DS_UUID)
    assert_target_ktk_readable(store, TGT_DS_UUID)


def test_copy_eager_with_rename_different_store(dataset_to_copy, store, store2):
    """
    Copies and renames DS between stores
    """
    copy_dataset(
        source_dataset_uuid=SRC_DS_UUID,
        target_dataset_uuid=TGT_DS_UUID,
        store=store,
        target_store=store2,
    )
    assert_target_keys(store, SRC_DS_UUID, store2, TGT_DS_UUID)
    assert_target_ktk_readable(store2, TGT_DS_UUID)


def test_copy_eager_no_target_uuid(dataset_to_copy, store, store2):
    copy_dataset(
        source_dataset_uuid=SRC_DS_UUID,
        target_dataset_uuid=None,
        store=store,
        target_store=store2,
    )
    assert_target_keys(store, SRC_DS_UUID, store2, SRC_DS_UUID)
    assert_target_ktk_readable(store2, SRC_DS_UUID)


def test_copy_eager_no_target_store(dataset_to_copy, store, store2):
    copy_dataset(
        source_dataset_uuid=SRC_DS_UUID,
        target_dataset_uuid=TGT_DS_UUID,
        store=store,
        target_store=None,
    )
    assert_target_keys(store, SRC_DS_UUID, store, TGT_DS_UUID)
    assert_target_ktk_readable(store, TGT_DS_UUID)


def test_copy_eager_without_rename_different_store(dataset_to_copy, store, store2):
    """
    Copies DS between stores while keeping the name
    """
    copy_dataset(
        source_dataset_uuid=SRC_DS_UUID,
        target_dataset_uuid=SRC_DS_UUID,
        store=store,
        target_store=store2,
    )
    assert_target_keys(store, SRC_DS_UUID, store2, SRC_DS_UUID)
    assert_target_ktk_readable(store2, SRC_DS_UUID)


def test_copy_same_source_and_target(dataset_to_copy, store):
    with pytest.raises(ValueError):
        copy_dataset(
            source_dataset_uuid=SRC_DS_UUID,
            target_dataset_uuid=SRC_DS_UUID,
            store=store,
            target_store=store,
        )
