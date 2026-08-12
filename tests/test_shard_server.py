import pytest
from fastapi.testclient import TestClient

from search.distribution.shard_builder import build_shards
from search.distribution.shard_index import ShardIndex
from search.distribution.shard_server import create_app
from search.ranking.bm25 import BM25Ranker

DUMP = "data/dumps/simplewiki.xml.bz2"
CORPUS = 500


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("single_shard_cluster")
    build_shards(DUMP, str(out), shard_count=1, limit=CORPUS)
    return f"{out}/shard-0"


@pytest.fixture
def client(shard_dir):
    with TestClient(create_app(shard=ShardIndex(shard_dir))) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats(client, shard_dir):
    idx = ShardIndex(shard_dir)
    try:
        response = client.get("/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["local_document_count"] == idx.document_count  # 1 shard: local == global
        assert body["global_document_count"] == idx.document_count
        assert body["vocabulary_size"] == idx.vocabulary_size
        assert body["avgdl"] == pytest.approx(idx.avgdl)
    finally:
        idx.close()


def test_search_returns_the_shape_the_coordinator_expects(client):
    response = client.get("/search", params={"q": "solar system", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) <= 5
    for result in body:
        assert set(result) == {"doc_id", "score", "title"}


def test_search_results_match_direct_bm25_ranking(client, shard_dir):
    """The route is a thin wrapper: it shouldn't change what BM25Ranker
    already computes, only serialize it."""
    idx = ShardIndex(shard_dir)
    try:
        expected = BM25Ranker(idx).search("world war", top_k=5)

        body = client.get("/search", params={"q": "world war", "k": 5}).json()
        actual = [(r["doc_id"], r["score"]) for r in body]

        assert actual == [(doc_id, pytest.approx(score)) for doc_id, score in expected]
    finally:
        idx.close()


def test_search_missing_query_is_422(client):
    assert client.get("/search").status_code == 422
