import pytest

from search.distribution.coordinator import Coordinator
from search.distribution.shard_builder import build_shards
from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
CORPUS = 2000


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    out = tmp_path_factory.mktemp("cluster")
    build_shards(DUMP, str(out), shard_count=3, limit=CORPUS)
    coord = Coordinator(str(out))
    yield coord
    coord.close()


@pytest.fixture(scope="module")
def single():
    index = InvertedIndex()
    for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=CORPUS), 1):
        index.add_document(doc_id, text, title=title)
    return BM25Ranker(index)


@pytest.mark.parametrize("query", [
    "solar system", "world war", "machine learning", "united states", "music",
])
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
