from typing import Callable, Union

from minimalkv import KeyValueStore

StoreFactory = Callable[[], KeyValueStore]
StoreInput = Union[str, KeyValueStore, StoreFactory]
