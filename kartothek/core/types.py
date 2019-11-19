from typing import Callable

from simplekv import KeyValueStore

from kartothek.serialization import (
    ConjunctionType,
    LiteralType,
    LiteralValue,
    PredicatesType,
)

StoreFactory = Callable[[], KeyValueStore]

__all__ = ["ConjunctionType", "LiteralType", "LiteralValue", "PredicatesType"]
