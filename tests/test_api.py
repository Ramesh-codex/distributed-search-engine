import pytest
from fastapi.testclient import TestClient

from search.api.app import create_app
from search.index.inverted_index import InvertedIndex


@pytest.fixture
def index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_document(1, "distributed systems are running",
                      url="https://example.com/1", title="Distributed Systems")
    idx.add_document(2, "distributed caches and running systems",
                      url="https://example.com/2", title="Caching")
    idx.add_document(3, "a completely unrelated document about gardening",
                      url="https://example.com/3", title="Gardening")
    return idx


@pytest.fixture
def client(index):
    with TestClient(create_app(index=index)) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_reflects_the_index(client, index):
    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == index.document_count
    assert body["vocabulary_size"] == index.vocabulary_size
    assert body["avgdl"] == pytest.approx(index.avgdl)


def test_home_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_search_returns_matching_docs_only(client):
    response = client.get("/search", params={"q": "distributed caching", "k": 5})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "distributed caching"
    assert body["total_hits"] == 2  # docs 1 and 2 share the "distribut"/"cach" stems
    doc_ids = {r["doc_id"] for r in body["results"]}
    assert doc_ids == {1, 2}
    assert 3 not in doc_ids  # the gardening doc shares no terms with the query


def test_search_results_are_sorted_by_score_descending(client):
    response = client.get("/search", params={"q": "distributed caching", "k": 5})
    scores = [r["score"] for r in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(client):
    response = client.get("/search", params={"q": "systems", "k": 1})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["total_hits"] == 2  # both doc 1 and doc 2 contain "systems", only 1 returned


def test_search_no_matches_returns_empty_results_not_an_error(client):
    response = client.get("/search", params={"q": "xyznonexistentterm"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["total_hits"] == 0


def test_search_missing_query_is_422(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_search_k_out_of_range_is_422(client):
    assert client.get("/search", params={"q": "systems", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "systems", "k": 101}).status_code == 422
