import time

from search.cache.lru import LRUCache
from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker


class CachedRanker:
    """BM25 ranker with an LRU in front of it.

    Caches the final ranked list keyed by (normalised query, top_k), not the
    per-term postings: postings are already O(1) to fetch, and the expensive
    part is scoring every document in them. Including top_k in the key matters
    because a cached k=10 result cannot satisfy a k=50 request.

    The index is immutable once built, so entries never go stale. A mutable
    index would need invalidation on every add_document.
    """

    def __init__(self, index: InvertedIndex, capacity: int = 1024) -> None:
        self._ranker = BM25Ranker(index)
        self._cache = LRUCache(capacity=capacity)

    @staticmethod
    def _key(query: str, top_k: int) -> str:
        # Normalising here means "Solar System", "solar system" and
        # " solar  system " share one entry instead of three.
        return f"{' '.join(query.lower().split())}|{top_k}"

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        key = self._key(query, top_k)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        results = self._ranker.search(query, top_k=top_k)
        self._cache.put(key, results)
        return results

    @property
    def hit_rate(self) -> float:
        return self._cache.hit_rate

    @property
    def stats(self) -> dict:
        return {


"hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": round(self._cache.hit_rate, 4),
            "entries": len(self._cache),
        }
