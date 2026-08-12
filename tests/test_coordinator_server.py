import httpx
import pytest
from fastapi.testclient import TestClient

from search.distribution.coordinator import Coordinator
from search.distribution.coordinator_server import create_app
from search.distribution.shard_builder import build_shards
from search.distribution.shard_index import ShardIndex
from search.ranking.bm25 import BM25Ranker

DUMP = "data/dumps/simplewiki.xml.bz2"
CORPUS = 500
SHARD_NAMES = ("shard-0", "shard-1", "shard-2")


def _shard_url(name: str) -> str:
    return f"http://{name}"


@pytest.fixture(scope="module")
def shard_rankers(tmp_path_factory):
    out = tmp_path_factory.mktemp("coordinator_server_cluster")
    build_shards(DUMP, str(out), shard_count=3, limit=CORPUS)
    rankers = {}
    for name in SHARD_NAMES:
        shard = ShardIndex(f"{out}/{name}")
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
def client(shard_rankers):
    coordinator = Coordinator(
        shard_urls=[_shard_url(n) for n in SHARD_NAMES],
        client=httpx.Client(transport=_mock_transport(shard_rankers)),
    )
    with TestClient(create_app(coordinator=coordinator)) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search(client):
    response = client.get("/search", params={"q": "solar system", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "solar system"
    assert body["degraded"] is False
    assert body["shards_queried"] == 3
    assert len(body["results"]) <= 5


def test_stats_lists_all_shards_closed_before_any_failure(client):
    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert set(body["shard_urls"]) == {_shard_url(n) for n in SHARD_NAMES}
    assert len(body["breakers"]) == 3
    for status in body["breakers"].values():
        assert status["state"] == "closed"
        assert status["consecutive_failures"] == 0


def test_stats_reflects_breaker_state_after_failures(shard_rankers):
    coordinator = Coordinator(
        shard_urls=[_shard_url(n) for n in SHARD_NAMES],
        client=httpx.Client(transport=_mock_transport(shard_rankers, down=frozenset({"shard-1"}))),
        breaker_failure_threshold=2,
    )
    with TestClient(create_app(coordinator=coordinator)) as c:
        c.get("/search", params={"q": "solar system"})
        c.get("/search", params={"q": "solar system"})

        body = c.get("/stats").json()
        assert body["breakers"][_shard_url("shard-1")]["state"] == "open"
        assert body["breakers"][_shard_url("shard-1")]["consecutive_failures"] == 2
        assert body["breakers"][_shard_url("shard-0")]["state"] == "closed"
