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
