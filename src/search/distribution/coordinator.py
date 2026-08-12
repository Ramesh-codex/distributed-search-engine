import heapq
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from search.distribution.shard_index import ShardIndex
from search.ranking.bm25 import BM25Ranker


class Coordinator:
    """Scatter-gather over shards, merging local top-K into a global top-K.

    Each shard returns its own top-K, already sorted. heapq.merge over those
    pre-sorted sequences is O(NK log N) and lazy, so taking the first K stops
    early; concatenating and sorting would be O(NK log NK) over results that
    are mostly discarded. This is the k-way merge, and pre-sortedness is what
    makes it worth doing.

    Shards are queried in parallel: they are independent, and scatter latency
    is bounded by the slowest shard rather than their sum. In-process here
    the parallelism is limited by the GIL, but the structure is the same one
    the HTTP version uses, where the work is I/O-bound and threads help fully.
    """

    def __init__(self, cluster_dir: str) -> None:
        self._dir = Path(cluster_dir)
        with open(self._dir / "cluster.json", encoding="utf-8") as handle:
            self._meta = json.load(handle)

        self._shards: dict[str, ShardIndex] = {}
        self._rankers: dict[str, BM25Ranker] = {}
        for name in self._meta["shards"]:
            shard = ShardIndex(str(self._dir / name))
            self._shards[name] = shard
            self._rankers[name] = BM25Ranker(shard)

        self._pool = ThreadPoolExecutor(max_workers=len(self._shards))
        self._down: set[str] = set()

    def _query_shard(self, name: str, query: str, top_k: int):
        if name in self._down:
            raise RuntimeError(f"{name} is marked down")
        results = self._rankers[name].search(query, top_k=top_k)
        return name, [(score, doc_id, name) for doc_id, score in results]

    def search(self, query: str, top_k: int = 10) -> dict:
        start = time.perf_counter()
        futures = {
            name: self._pool.submit(self._query_shard, name, query, top_k)
            for name in self._shards
        }

        per_shard = []
        failed = []
        for name, future in futures.items():
            try:
                _, scored = future.result(timeout=5.0)
                per_shard.append(scored)
            except Exception:
                # Partial results beat no results: two thirds of the corpus
                # is a usable answer, provided the caller is told it is
                # partial. The degraded flag is what makes that honest.
                failed.append(name)

        merged = heapq.merge(*per_shard, reverse=True)
        top = []
        for score, doc_id, shard in merged:
            top.append({
                "doc_id": doc_id,
                "score": score,
                "shard": shard,
                "title": self._shards[shard].document(doc_id).title,
            })
            if len(top) == top_k:
                break

        return {
            "query": query,
            "results": top,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
            "shards_queried": len(per_shard),
            "shards_failed": failed,
            "degraded": bool(failed),
        }

    def kill_shard(self, name: str) -> None:
        """Simulated failure, for the resilience benchmark."""
        self._down.add(name)

    def revive_shard(self, name: str) -> None:
        self._down.discard(name)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for shard in self._shards.values():
            shard.close()
