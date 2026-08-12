from search.distribution.consistent_hash import ConsistentHashRing

SHARDS = ["shard-0", "shard-1", "shard-2"]
KEYS = [f"doc:{i}" for i in range(30000)]
fair = len(KEYS) / len(SHARDS)

print("| replicas | shard-0 | shard-1 | shard-2 | max deviation |")
print("|---|---|---|---|---|")
for replicas in (1, 10, 50, 150, 500, 1000):
    ring = ConsistentHashRing(SHARDS, replicas=replicas)
    counts = {s: 0 for s in SHARDS}
    for key in KEYS:
        counts[ring.get_shard(key)] += 1
    worst = max(abs(c - fair) / fair for c in counts.values())
    print(f"| {replicas} | " + " | ".join(str(counts[s]) for s in SHARDS)
          + f" | {worst:.1%} |")
