from kartothek.core.factory import DatasetFactory


def test_repr(store_factory):
    factory = DatasetFactory(
        dataset_uuid="dataset_uuid", store_factory=store_factory  # does not exist
    )
    assert repr(factory) == "<DatasetFactory: uuid=dataset_uuid is_loaded=False>"
