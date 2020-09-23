from typing import Callable, Union

from simplekv import KeyValueStore

StoreFactory = Callable[[], KeyValueStore]
StoreInput = Union[str, KeyValueStore, StoreFactory]
