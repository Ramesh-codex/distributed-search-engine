import pytest

from search.cache.lru import LRUCache


def test_evicts_least_recently_used():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_read_refreshes_recency():
    """The property that separates LRU from FIFO: reading 'a' must save it."""
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_overwrite_does_not_grow():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert len(cache) == 1
    assert cache.get("a") == 2


def test_hit_rate():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.get("a")
    cache.get("missing")
    assert cache.hit_rate == 0.5


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
