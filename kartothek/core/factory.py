# -*- coding: utf-8 -*-

import copy
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

from kartothek.core.dataset import DatasetMetadata, DatasetMetadataBase
from kartothek.core.typing import StoreInput
from kartothek.core.utils import lazy_store

if TYPE_CHECKING:
    from simplekv import KeyValueStore

T = TypeVar("T", bound="DatasetFactory")


class DatasetFactory(DatasetMetadataBase):
    """
    Container holding metadata caching storage access.
    """

    _nullable_attributes = ["_cache_metadata", "_cache_store"]

    def __init__(
        self,
        dataset_uuid: str,
        store_factory: StoreInput,
        load_schema: bool = True,
        load_all_indices: bool = False,
        load_dataset_metadata: bool = True,
    ) -> None:
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
        self._cache_metadata: Optional[DatasetMetadata] = None
        self._cache_store = None

        self.store_factory = lazy_store(store_factory)
        self.dataset_uuid = dataset_uuid
        self.load_schema = load_schema
        self._ds_callable = None
        self.is_loaded = False
        self.load_dataset_metadata = load_dataset_metadata
        self.load_all_indices_flag = load_all_indices

    def __repr__(self):
        return "<DatasetFactory: uuid={} is_loaded={}>".format(
            self.dataset_uuid, self.is_loaded
        )

    @property
    def store(self) -> "KeyValueStore":
        if self._cache_store is None:
            self._cache_store = self.store_factory()
        return self._cache_store

    def _instantiate_metadata_cache(self: T) -> T:
        if self._cache_metadata is None:
            if self._ds_callable:
                # backwards compat
                self._cache_metadata = self._ds_callable()
            else:
                self._cache_metadata = DatasetMetadata.load_from_store(
                    uuid=self.dataset_uuid,
                    store=self.store,
                    load_schema=self.load_schema,
                    load_all_indices=self.load_all_indices_flag,
                )
            if not self.load_dataset_metadata:
                self._cache_metadata.metadata = {}
        self.is_loaded = True
        return self

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        self._instantiate_metadata_cache()
        # The above line ensures non-None
        return cast(DatasetMetadata, self._cache_metadata)

    def invalidate(self) -> None:
        self.is_loaded = False
        self._cache_metadata = None
        self._cache_store = None

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

    def __deepcopy__(self, memo) -> "DatasetFactory":
        new_obj = DatasetFactory(
            dataset_uuid=self.dataset_uuid,
            store_factory=self.store_factory,
            load_schema=self.load_schema,
            load_all_indices=self.load_all_indices_flag,
        )
        if self._cache_metadata is not None:
            new_obj._cache_metadata = copy.deepcopy(self._cache_metadata)
        return new_obj

    def load_index(self: T, column, store=None) -> T:
        self._cache_metadata = self.dataset_metadata.load_index(column, self.store)
        return self

    def load_all_indices(
        self: T, store: Any = None, load_partition_indices: bool = True,
    ) -> T:
        self._cache_metadata = self.dataset_metadata.load_all_indices(
            self.store, load_partition_indices=load_partition_indices
        )
        return self

    def load_partition_indices(self: T) -> T:
        self._cache_metadata = self.dataset_metadata.load_partition_indices()
        return self


def _ensure_factory(
    dataset_uuid: Optional[str],
    store: StoreInput,
    factory: Optional[DatasetFactory],
    load_dataset_metadata: bool,
    load_schema: bool = True,
) -> DatasetFactory:

    if store is None and dataset_uuid is None and factory is not None:
        return factory
    elif store is not None and dataset_uuid is not None and factory is None:
        return DatasetFactory(
            dataset_uuid=dataset_uuid,
            store_factory=lazy_store(store),
            load_dataset_metadata=load_dataset_metadata,
            load_schema=load_schema,
        )

    else:
        raise ValueError(
            "Need to supply either a `factory` or `dataset_uuid` and `store`"
        )
