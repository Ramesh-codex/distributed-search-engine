import time

import httpx
import pytest

from search.distribution.coordinator import Coordinator, LocalCoordinator
from search.distribution.shard_builder import build_shards
from search.distribution.shard_index import ShardIndex
from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
CORPUS = 2000
SHARD_NAMES = ("shard-0", "shard-1", "shard-2")

QUERIES = ["solar system", "world war", "machine learning", "united states", "music"]


@pytest.fixture(scope="module")
def cluster_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("cluster")
    build_shards(DUMP, str(out), shard_count=3, limit=CORPUS)
    return str(out)


@pytest.fixture(scope="module")
def cluster(cluster_dir):
    coord = LocalCoordinator(cluster_dir)
    yield coord
    coord.close()


@pytest.fixture(scope="module")
def single_index():
    index = InvertedIndex()
    for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=CORPUS), 1):
        index.add_document(doc_id, text, title=title)
    return index


@pytest.fixture(scope="module")
def single(single_index):
    return BM25Ranker(single_index)


@pytest.mark.parametrize("query", QUERIES)
def test_sharded_ranking_matches_single_node(cluster, single, query):
    """The correctness property of the whole design: partitioning the corpus
    must not change the ranking. Holds only because global df, N and avgdl
    are shipped to every shard -- with local statistics these diverge."""
    expected = single.search(query, top_k=10)
    actual = [(r["doc_id"], r["score"]) for r in cluster.search(query, top_k=10)["results"]]
    assert [d for d, _ in expected] == [d for d, _ in actual]
    for (_, a), (_, b) in zip(expected, actual):
        assert a == pytest.approx(b, rel=1e-12)


def test_degrades_gracefully_when_a_shard_is_down(cluster):
    """Partial results beat no results, provided the caller is told."""
    cluster.kill_shard("shard-1")
    try:
        out = cluster.search("solar system", top_k=10)
        assert out["degraded"] is True
        assert out["shards_failed"] == ["shard-1"]
        assert out["shards_queried"] == 2
        assert len(out["results"]) > 0
        assert all(r["shard"] != "shard-1" for r in out["results"])
    finally:
        cluster.revive_shard("shard-1")


# --- HTTP Coordinator -------------------------------------------------------
#
# Same correctness and degraded-flag properties as above, but exercised over
# Coordinator's HTTP path instead of LocalCoordinator's in-process one. An
# httpx.MockTransport stands in for the shard_server processes: it routes
# each request to the real ShardIndex/BM25Ranker for the URL's host and
# returns exactly the JSON shape shard_server's /search route returns, so
# these tests cover Coordinator's request building, response parsing, and
# merge logic without a container in sight.

def _shard_url(name: str) -> str:
    return f"http://{name}"


@pytest.fixture(scope="module")
def shard_rankers(cluster_dir):
    rankers: dict[str, tuple[ShardIndex, BM25Ranker]] = {}
    for name in SHARD_NAMES:
        shard = ShardIndex(f"{cluster_dir}/{name}")
        rankers[name] = (shard, BM25Ranker(shard))
    yield rankers
    for shard, _ in rankers.values():
        shard.close()


def _mock_transport(shard_rankers, down: frozenset[str] = frozenset()) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.host
        if name in down:
            return httpx.Response(503, json={"detail": f"{name} unavailable"})

        shard, ranker = shard_rankers[name]
        params = request.url.params
        hits = ranker.search(params["q"], top_k=int(params.get("k", "10")))
        body = [
            {"doc_id": doc_id, "score": score, "title": shard.document(doc_id).title}
            for doc_id, score in hits
        ]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.fixture
def http_cluster(shard_rankers):
    client = httpx.Client(transport=_mock_transport(shard_rankers))
    coord = Coordinator(shard_urls=[_shard_url(n) for n in SHARD_NAMES], client=client)
    yield coord
    coord.close()


@pytest.mark.parametrize("query", QUERIES)
def test_http_coordinator_matches_single_node(http_cluster, single, query):
    expected = single.search(query, top_k=10)
    actual = [(r["doc_id"], r["score"]) for r in http_cluster.search(query, top_k=10)["results"]]
    assert [d for d, _ in expected] == [d for d, _ in actual]
    for (_, a), (_, b) in zip(expected, actual):
        assert a == pytest.approx(b, rel=1e-12)


def test_http_coordinator_carries_titles_from_shard_responses(http_cluster, single_index):
    actual = http_cluster.search("solar system", top_k=1)["results"]
    assert actual[0]["title"] == single_index.document(actual[0]["doc_id"]).title


def test_http_coordinator_degrades_when_a_shard_returns_an_error(shard_rankers):
    client = httpx.Client(transport=_mock_transport(shard_rankers, down=frozenset({"shard-1"})))
    coord = Coordinator(shard_urls=[_shard_url(n) for n in SHARD_NAMES], client=client)
    try:
        out = coord.search("solar system", top_k=10)
        assert out["degraded"] is True
        assert out["shards_failed"] == [_shard_url("shard-1")]
        assert out["shards_queried"] == 2
        assert len(out["results"]) > 0
        assert all(r["shard"] != _shard_url("shard-1") for r in out["results"])
    finally:
        coord.close()


def test_http_coordinator_kill_shard_marks_it_down_without_a_request(shard_rankers):
    """kill_shard short-circuits before any HTTP call: the transport should
    never even see a request for the killed shard."""
    requested_hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        shard, ranker = shard_rankers[request.url.host]
        hits = ranker.search(request.url.params["q"], top_k=10)
        body = [
            {"doc_id": doc_id, "score": score, "title": shard.document(doc_id).title}
            for doc_id, score in hits
        ]
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    coord = Coordinator(shard_urls=[_shard_url(n) for n in SHARD_NAMES], client=client)
    try:
        coord.kill_shard(_shard_url("shard-1"))
        out = coord.search("solar system", top_k=10)
        assert out["degraded"] is True
        assert _shard_url("shard-1") not in requested_hosts
    finally:
        coord.close()


def test_coordinator_reads_shard_urls_from_env(monkeypatch):
    monkeypatch.setenv("SHARD_URLS", " http://shard-0 ,http://shard-1,http://shard-2 ")
    coord = Coordinator()
    try:
        assert coord._shard_urls == ["http://shard-0", "http://shard-1", "http://shard-2"]
    finally:
        coord.close()


def test_coordinator_requires_shard_urls():
    with pytest.raises(ValueError):
        Coordinator(shard_urls=[])


def test_coordinator_default_timeouts_split_connect_and_read():
    coord = Coordinator(shard_urls=["http://shard-0"])
    try:
        assert coord._request_timeout.connect == pytest.approx(0.2)
        assert coord._request_timeout.read == pytest.approx(2.0)
    finally:
        coord.close()


def test_coordinator_timeouts_are_configurable():
    coord = Coordinator(shard_urls=["http://shard-0"], connect_timeout=0.5, read_timeout=3.0)
    try:
        assert coord._request_timeout.connect == pytest.approx(0.5)
        assert coord._request_timeout.read == pytest.approx(3.0)
    finally:
        coord.close()


def test_coordinator_breaker_thresholds_are_configurable():
    coord = Coordinator(
        shard_urls=["http://shard-0"], breaker_failure_threshold=7, breaker_cooldown_seconds=42.0,
    )
    try:
        breaker = coord._breakers["http://shard-0"]
        assert breaker._failure_threshold == 7
        assert breaker._cooldown_seconds == 42.0
    finally:
        coord.close()


# --- Circuit breaker, wired through Coordinator -----------------------------
#
# CircuitBreaker's own state-machine transitions are covered in isolation by
# test_circuit_breaker.py. These tests instead confirm Coordinator actually
# wires a breaker in front of every shard call correctly: it opens from real
# consecutive HTTP failures, an open breaker stops _query_shard from making
# a request at all (not just from succeeding), and the half-open probe both
# recovers and re-fails correctly through the real search() path.

def _counting_flaky_transport(shard_rankers, flaky: str, healthy_after: int | None = None):
    """`flaky` fails every request until `healthy_after` calls have been
    made to it, then serves normally (None: always fails). Returns the
    transport plus a per-shard call counter to assert against."""
    call_counts = {name: 0 for name in shard_rankers}

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.host
        call_counts[name] += 1
        if name == flaky and (healthy_after is None or call_counts[name] <= healthy_after):
            return httpx.Response(503, json={"detail": f"{name} unavailable"})
        shard, ranker = shard_rankers[name]
        params = request.url.params
        hits = ranker.search(params["q"], top_k=int(params.get("k", "10")))
        body = [
            {"doc_id": doc_id, "score": score, "title": shard.document(doc_id).title}
            for doc_id, score in hits
        ]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), call_counts


def test_coordinator_breaker_opens_after_threshold_and_then_skips_the_shard(shard_rankers):
    transport, call_counts = _counting_flaky_transport(shard_rankers, flaky="shard-1")
    coord = Coordinator(
        shard_urls=[_shard_url(n) for n in SHARD_NAMES],
        client=httpx.Client(transport=transport),
        breaker_failure_threshold=3,
        breaker_cooldown_seconds=10.0,
    )
    try:
        for _ in range(3):
            out = coord.search("solar system", top_k=5)
            assert out["degraded"] is True
        assert call_counts["shard-1"] == 3
        assert coord.breaker_stats[_shard_url("shard-1")]["state"] == "open"

        # 4th call: breaker is open and the 10s cooldown hasn't elapsed, so
        # shard-1 must be skipped without ever reaching the transport.
        out = coord.search("solar system", top_k=5)
        assert call_counts["shard-1"] == 3
        assert out["shards_failed"] == [_shard_url("shard-1")]
        assert out["degraded"] is True
    finally:
        coord.close()


def test_coordinator_breaker_half_open_probe_closes_on_success(shard_rankers):
    transport, call_counts = _counting_flaky_transport(shard_rankers, flaky="shard-1", healthy_after=2)
    coord = Coordinator(
        shard_urls=[_shard_url(n) for n in SHARD_NAMES],
        client=httpx.Client(transport=transport),
        breaker_failure_threshold=2,
        breaker_cooldown_seconds=0.05,
    )
    try:
        coord.search("solar system", top_k=5)
        coord.search("solar system", top_k=5)
        assert coord.breaker_stats[_shard_url("shard-1")]["state"] == "open"

        time.sleep(0.1)  # let the cooldown elapse; shard-1 is "healthy" again by now

        out = coord.search("solar system", top_k=5)
        assert call_counts["shard-1"] == 3  # 2 failures + the half-open probe
        assert out["degraded"] is False
        assert coord.breaker_stats[_shard_url("shard-1")]["state"] == "closed"
        assert coord.breaker_stats[_shard_url("shard-1")]["consecutive_failures"] == 0
    finally:
        coord.close()


def test_coordinator_breaker_reopens_on_a_failed_probe(shard_rankers):
    transport, call_counts = _counting_flaky_transport(shard_rankers, flaky="shard-1")  # never recovers
    coord = Coordinator(
        shard_urls=[_shard_url(n) for n in SHARD_NAMES],
        client=httpx.Client(transport=transport),
        breaker_failure_threshold=2,
        breaker_cooldown_seconds=0.05,
    )
    try:
        coord.search("solar system", top_k=5)
        coord.search("solar system", top_k=5)
        assert coord.breaker_stats[_shard_url("shard-1")]["state"] == "open"

        time.sleep(0.1)
        out = coord.search("solar system", top_k=5)  # the probe: still failing
        assert call_counts["shard-1"] == 3
        assert out["degraded"] is True
        assert coord.breaker_stats[_shard_url("shard-1")]["state"] == "open"

        # Immediately after: back within the fresh cooldown, no further request.
        coord.search("solar system", top_k=5)
        assert call_counts["shard-1"] == 3
    finally:
        coord.close()
