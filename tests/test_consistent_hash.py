import pytest

from search.distribution.consistent_hash import ConsistentHashRing

SHARDS = ["shard-0", "shard-1", "shard-2"]
KEYS = [f"doc:{i}" for i in range(30000)]


def test_assignment_is_deterministic():
    a = ConsistentHashRing(SHARDS)
    b = ConsistentHashRing(SHARDS)
    for key in KEYS[:500]:
        assert a.get_shard(key) == b.get_shard(key)


def test_distribution_is_roughly_even():
    """1000 virtual nodes should keep every shard within 3% of fair share.
    With one point per shard this test fails regularly -- that is the whole
    argument for virtual nodes."""
    ring = ConsistentHashRing(SHARDS)
    counts = {s: 0 for s in SHARDS}
    for key in KEYS:
        counts[ring.get_shard(key)] += 1
    fair = len(KEYS) / len(SHARDS)
    for shard, count in counts.items():
        assert abs(count - fair) / fair < 0.03, (shard, count, counts)


def test_removing_a_shard_moves_only_its_own_keys():
    """The property modulo hashing does not have. Removing 1 of 3 shards
    should reassign about a third of keys -- and critically, no key that
    lived on a surviving shard may move."""
    ring = ConsistentHashRing(SHARDS)
    before = {key: ring.get_shard(key) for key in KEYS}

    ring.remove_shard("shard-2")
    after = {key: ring.get_shard(key) for key in KEYS}

    moved = [k for k in KEYS if before[k] != after[k]]
    assert all(before[k] == "shard-2" for k in moved)

    fraction = len(moved) / len(KEYS)
    assert 0.28 < fraction < 0.39, fraction


def test_adding_a_shard_moves_only_a_fair_share():
    ring = ConsistentHashRing(SHARDS)
    before = {key: ring.get_shard(key) for key in KEYS}

    ring.add_shard("shard-3")
    after = {key: ring.get_shard(key) for key in KEYS}

    moved = [k for k in KEYS if before[k] != after[k]]
    assert all(after[k] == "shard-3" for k in moved)

    fraction = len(moved) / len(KEYS)
    assert 0.20 < fraction < 0.31, fraction


def test_rejects_empty_ring():
    with pytest.raises(ValueError):
        ConsistentHashRing([])
