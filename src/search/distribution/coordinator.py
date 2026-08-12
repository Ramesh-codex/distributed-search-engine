import heapq
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from search.distribution.circuit_breaker import CircuitBreaker, CircuitOpenError
from search.distribution.shard_index import ShardIndex
from search.ranking.bm25 import BM25Ranker

DEFAULT_TIMEOUT_SECONDS = 5.0

# A shard on the same network either answers in milliseconds or is not
# there. Connect and read are split because they fail differently: a dead
# host or a firewalled port hangs on connect, while a slow-but-alive shard
# hangs on read -- collapsing both into one timeout means either the
# healthy case has to tolerate the dead-host number, or the dead-host case
# has to wait as long as a legitimately slow query is allowed to take.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 0.2
DEFAULT_READ_TIMEOUT_SECONDS = 2.0

DEFAULT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_BREAKER_COOLDOWN_SECONDS = 10.0


class LocalCoordinator:
    """Scatter-gather over shards, merging local top-K into a global top-K.

    In-process version: shards are ShardIndex objects opened directly off
    disk, no network involved. Kept alongside the HTTP Coordinator so
    tests/test_sharding.py can exercise the merge/degraded-flag logic
    without containers.

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
                _, scored = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
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


class Coordinator:
    """Scatter-gather over shard HTTP services, merging local top-K into a
    global top-K.

    Same structure as LocalCoordinator -- same k-way heapq.merge, same
    thread-pool scatter with a per-shard result timeout, same degraded-flag
    semantics on partial failure. The only thing that changed is how a
    shard is asked for its top-K: an HTTP GET instead of an in-process
    BM25Ranker.search() call.

    Shards are identified by URL, not by name: unlike LocalCoordinator this
    class never reads cluster.json (it has no filesystem access to a shard's
    directory at all), so the URL passed in SHARD_URLS is the only handle it
    has. `shards_failed` and each result's `shard` field are URLs.

    Titles travel with each shard's response instead of being looked up
    afterward, because there is no local ShardIndex to look them up from
    once results are merged -- the whole point of going over HTTP is that
    the shard's data lives in another process.

    A CircuitBreaker per shard URL sits in front of the HTTP call. Without
    it, a genuinely down shard pays its full connect+read timeout on every
    single query, forever -- the timeout bounds one request, not the
    decision to keep sending them. After enough consecutive failures the
    breaker opens and _query_shard raises immediately, no network call
    attempted, until the cooldown elapses and one probe is let through.
    """

    def __init__(
        self,
        shard_urls: list[str] | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        client: httpx.Client | None = None,
        breaker_failure_threshold: int | None = None,
        breaker_cooldown_seconds: float | None = None,
    ) -> None:
        """Every tunable falls back to an environment variable, then to the
        module default, in that order -- so docker-compose can configure a
        deployed coordinator without a code change, while tests pass exact
        values directly and never touch the environment.
        """
        if shard_urls is None:
            raw = os.environ.get("SHARD_URLS", "")
            shard_urls = [url.strip() for url in raw.split(",") if url.strip()]
        if not shard_urls:
            raise ValueError("no shard URLs configured: pass shard_urls or set SHARD_URLS")

        if connect_timeout is None:
            connect_timeout = float(os.environ.get("SHARD_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_SECONDS))
        if read_timeout is None:
            read_timeout = float(os.environ.get("SHARD_READ_TIMEOUT", DEFAULT_READ_TIMEOUT_SECONDS))
        if breaker_failure_threshold is None:
            breaker_failure_threshold = int(
                os.environ.get("BREAKER_FAILURE_THRESHOLD", DEFAULT_BREAKER_FAILURE_THRESHOLD)
            )
        if breaker_cooldown_seconds is None:
            breaker_cooldown_seconds = float(
                os.environ.get("BREAKER_COOLDOWN_SECONDS", DEFAULT_BREAKER_COOLDOWN_SECONDS)
            )

        self._shard_urls = shard_urls
        self._request_timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout,
        )
        # The thread-pool result guard has to be strictly looser than the
        # request-level timeout, or it could fire first and mask the real
        # (informative) httpx timeout error with a generic "future timed out".
        self._future_timeout = connect_timeout + read_timeout + 1.0
        self._client = client if client is not None else httpx.Client()
        self._pool = ThreadPoolExecutor(max_workers=len(shard_urls))
        self._down: set[str] = set()
        self._breakers = {
            url: CircuitBreaker(
                failure_threshold=breaker_failure_threshold,
                cooldown_seconds=breaker_cooldown_seconds,
            )
            for url in shard_urls
        }

    def _query_shard(self, url: str, query: str, top_k: int):
        if url in self._down:
            raise RuntimeError(f"{url} is marked down")

        breaker = self._breakers[url]
        if not breaker.allow_request():
            raise CircuitOpenError(f"{url} circuit is open")

        try:
            response = self._client.get(
                f"{url}/search", params={"q": query, "k": top_k}, timeout=self._request_timeout,
            )
            response.raise_for_status()
        except Exception:
            breaker.record_failure()
            raise

        breaker.record_success()
        payload = response.json()
        return url, [(r["score"], r["doc_id"], url, r["title"]) for r in payload]

    def search(self, query: str, top_k: int = 10) -> dict:
        start = time.perf_counter()
        futures = {
            url: self._pool.submit(self._query_shard, url, query, top_k)
            for url in self._shard_urls
        }

        per_shard = []
        failed = []
        for url, future in futures.items():
            try:
                _, scored = future.result(timeout=self._future_timeout)
                per_shard.append(scored)
            except Exception:
                # Partial results beat no results: two thirds of the corpus
                # is a usable answer, provided the caller is told it is
                # partial. The degraded flag is what makes that honest.
                failed.append(url)

        merged = heapq.merge(*per_shard, reverse=True)
        top = []
        for score, doc_id, shard, title in merged:
            top.append({
                "doc_id": doc_id,
                "score": score,
                "shard": shard,
                "title": title,
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

    def kill_shard(self, url: str) -> None:
        """Simulated failure, for the resilience benchmark."""
        self._down.add(url)

    def revive_shard(self, url: str) -> None:
        self._down.discard(url)

    @property
    def shard_urls(self) -> list[str]:
        return list(self._shard_urls)

    @property
    def breaker_stats(self) -> dict[str, dict]:
        return {
            url: {"state": breaker.state, "consecutive_failures": breaker.consecutive_failures}
            for url, breaker in self._breakers.items()
        }

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self._client.close()
