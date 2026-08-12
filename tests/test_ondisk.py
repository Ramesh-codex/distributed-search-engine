from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.ondisk import OnDiskIndex, OnDiskIndexWriter


def test_ondisk_matches_in_memory(tmp_path):
    """The on-disk index must be a drop-in for the in-memory one: same
    postings, same df, same avgdl, and therefore identical BM25 scores."""
    memory = InvertedIndex()
    memory.add_document(1, "distributed machine learning systems", title="A")
    memory.add_document(2, "machine learning at scale", title="B")
    memory.add_document(3, "sourdough bread baking", title="C")

    OnDiskIndexWriter(str(tmp_path)).write(memory)
    disk = OnDiskIndex(str(tmp_path))

    try:
        assert disk.document_count == memory.document_count
        assert disk.vocabulary_size == memory.vocabulary_size
        assert disk.avgdl == memory.avgdl

        for term in ["machin", "learn", "distribut", "sourdough", "absent"]:
            assert disk.postings(term) == memory.postings(term), term
            assert disk.document_frequency(term) == memory.document_frequency(term)

        query = "distributed machine learning"
        assert BM25Ranker(disk).search(query) == BM25Ranker(memory).search(query)
    finally:
        disk.close()
