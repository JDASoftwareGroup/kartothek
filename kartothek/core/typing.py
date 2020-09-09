from typing import Callable, Union

from simplekv import KeyValueStore

STORE_TYPE = Union[str, KeyValueStore, Callable]
STORE_FACTORY_TYPE = Callable[[], KeyValueStore]
