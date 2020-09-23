from typing import Callable, Union

from simplekv import KeyValueStore

StoreInput = Union[str, KeyValueStore, Callable]
StoreFactory = Callable[[], KeyValueStore]
