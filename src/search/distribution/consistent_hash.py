import bisect
import hashlib


class ConsistentHashRing:
    """Maps keys to shards such that adding or removing a shard moves only
    K/N of the keys, not all of them.

    Modulo hashing (hash(key) % shard_count) is simpler and wrong for this:
    changing the shard count remaps almost every key, which for a search
    index means rebuilding the entire corpus to add one node. A ring assigns
    each key to the next shard clockwise, so removing a shard only reassigns
    that shard's arc.

    Virtual nodes exist because a ring with one point per shard distributes
    badly -- three random points on a circle rarely produce three equal arcs.
    Placing 150 points per shard averages the arcs out; imbalance falls
    roughly as 1/sqrt(replicas).
    """

    def __init__(self, shards: list[str], replicas: int = 1000) -> None:
        if not shards:
            raise ValueError("need at least one shard")
        self._replicas = replicas
        self._ring: dict[int, str] = {}
        self._sorted_keys: list[int] = []
        for shard in shards:
            self.add_shard(shard)

    @staticmethod
    def _hash(key: str) -> int:
        """md5 rather than Python's hash(): hash() is randomised per process
        by PYTHONHASHSEED, so the same doc would land on a different shard
        after a restart. Ring placement must be stable across processes."""
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_shard(self, shard: str) -> None:
        for i in range(self._replicas):
            point = self._hash(f"{shard}#{i}")
            self._ring[point] = shard
            bisect.insort(self._sorted_keys, point)

    def remove_shard(self, shard: str) -> None:
        for i in range(self._replicas):
            point = self._hash(f"{shard}#{i}")
            if point in self._ring:
                del self._ring[point]
                index = bisect.bisect_left(self._sorted_keys, point)
                self._sorted_keys.pop(index)

    def get_shard(self, key: str) -> str:
        """First ring point clockwise from the key's hash.

        bisect_right on a sorted list is O(log n); wrapping to index 0 when
        the key hashes past the last point is what makes it a ring.
        """
        if not self._sorted_keys:
            raise ValueError("ring is empty")
        point = self._hash(key)
        index = bisect.bisect_right(self._sorted_keys, point)
        if index == len(self._sorted_keys):
            index = 0
        return self._ring[self._sorted_keys[index]]

    @property
    def shards(self) -> list[str]:
        return sorted(set(self._ring.values()))
