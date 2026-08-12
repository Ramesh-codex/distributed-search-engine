from collections import OrderedDict
from typing import Any


class LRUCache:
    """Fixed-capacity LRU cache with O(1) get and put.

    OrderedDict is a hash map with a doubly-linked list threaded through it,
    which is exactly the structure the classic hand-rolled answer builds: the
    dict gives O(1) lookup, the list gives O(1) reordering and eviction.
    A plain dict has no ordering; a list has O(n) removal. This has both.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """A hit both returns the value and marks it most-recently-used.

        Reading without reordering would make this a FIFO cache, not an LRU --
        a frequently-read entry would still be evicted on age alone.
        """
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._capacity:
            # last=False pops the front, which is the least-recently-used end.
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        """The metric that justifies the cache existing. Reported alongside
        latency percentiles: a cache with a 3% hit rate is pure overhead."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0
