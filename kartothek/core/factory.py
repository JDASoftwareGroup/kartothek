# -*- coding: utf-8 -*-


import copy

from kartothek.core.dataset import DatasetMetadata, DatasetMetadataBase
from kartothek.core.utils import _check_callable


def _ensure_factory(
    dataset_uuid, store, factory, load_dataset_metadata, load_schema=True
):
    if store is None and dataset_uuid is None and factory is not None:
        return factory
    elif store is not None and dataset_uuid is not None and factory is None:
        return DatasetFactory(
            dataset_uuid=dataset_uuid,
            store_factory=store,
            load_dataset_metadata=load_dataset_metadata,
            load_schema=load_schema,
        )

    else:
        raise ValueError(
            "Need to supply either a `factory` or `dataset_uuid` and `store`"
        )


class DatasetFactory(DatasetMetadataBase):

    _nullable_attributes = ["_cache_metadata", "_cache_store"]

    def __init__(
        self,
        dataset_uuid,
        store_factory,
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=True,
    ):
        """
        A dataset factory object which can be used to cache dataset load operations. This class should be the primary user entry point when
        reading datasets.

        Example using the eager backend:

        .. code::

            from functools import partial
            from storefact import get_store_from_url
            from kartothek.io.eager import read_table

            ds_factory = DatasetFactory(
                dataset_uuid="my_test_dataset",
                store=partial(get_store_from_url, store_url)
            )

            df = read_table(factory=ds_factory)

        Parameters
        ----------
        dataset_uuid: str
            The unique indetifier for the dataset.
        store_factory: callable
            A callable which creates a KeyValueStore object
        load_schema: bool
            Load the schema information immediately.
        load_all_indices: bool
            Load all indices immediately.
        load_dataset_metadata: bool
            Keep the user metadata in memory
        """
        self._cache_metadata = None
        self._cache_store = None

        _check_callable(store_factory)
        self.store_factory = store_factory
        self.dataset_uuid = dataset_uuid
        self.load_schema = load_schema
        self._ds_callable = None
        self.is_loaded = False
        self.load_dataset_metadata = load_dataset_metadata
        self.load_all_indices_flag = load_all_indices
        self._exists = None

    def __repr__(self):
        return "<DatasetFactory: uuid={} is_loaded={}>".format(
            self.dataset_uuid, self.is_loaded
        )

    @property
    def exists(self):
        if self._exists is None:
            self._instantiate_metadata_cache()
        return self._exists

    @property
    def store(self):
        if self._cache_store is None:
            self._cache_store = self.store_factory()
        return self._cache_store

    def _instantiate_metadata_cache(self):
        if self._cache_metadata is None:
            if self._ds_callable:
                # backwards compat
                self._cache_metadata = self._ds_callable()
            else:
                try:
                    self._cache_metadata = DatasetMetadata.load_from_store(
                        uuid=self.dataset_uuid,
                        store=self.store,
                        load_schema=self.load_schema,
                        load_all_indices=self.load_all_indices_flag,
                    )
                    self._exists = True
                except KeyError:
                    self._exists = False
                    return self
            if not self.load_dataset_metadata:
                self._cache_metadata.metadata = {}
        self.is_loaded = True
        return self

    @property
    def dataset_metadata(self):
        if self._exists is not False:
            self._instantiate_metadata_cache()
        return self._cache_metadata

    def invalidate(self):
        self.is_loaded = False
        self._cache_metadata = None
        self._cache_store = None
        self._exists = None

    def __getattr__(self, name):
        # __getattr__ should only be called if the attribute cannot be found. if the
        # attribute is None, it still falls back to this call
        if name in self._nullable_attributes:
            return object.__getattribute__(self, name)
        self._instantiate_metadata_cache()
        ds = getattr(self, "dataset_metadata")
        return getattr(ds, name)

    def __getstate__(self):
        # remove cache
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_cache_")}

    def __setstate__(self, state):
        self.__init__(
            dataset_uuid=state["dataset_uuid"],
            store_factory=state["store_factory"],
            load_schema=state["load_schema"],
            load_all_indices=state["load_all_indices_flag"],
        )

    def __deepcopy__(self, memo):
        new_obj = DatasetFactory(
            dataset_uuid=self.dataset_uuid,
            store_factory=self.store_factory,
            load_schema=self.load_schema,
            load_all_indices=self.load_all_indices_flag,
        )
        if self._cache_metadata is not None:
            new_obj._cache_metadata = copy.deepcopy(self._cache_metadata)
        return new_obj

    def load_index(self, column, store=None):
        self._cache_metadata = self.dataset_metadata.load_index(column, self.store)
        return self

    def load_all_indices(self, load_partition_indices=True, store=None):
        self._cache_metadata = self.dataset_metadata.load_all_indices(
            self.store, load_partition_indices=load_partition_indices
        )
        return self

    def load_partition_indices(self):
        self._cache_metadata = self.dataset_metadata.load_partition_indices()
        return self
