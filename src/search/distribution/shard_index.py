import json
from pathlib import Path

from search.storage.ondisk import OnDiskIndex


class ShardIndex(OnDiskIndex):
    """An on-disk shard that reports global corpus statistics.

    BM25Ranker reads document_count, document_frequency and avgdl to compute
    IDF and length normalisation. By overriding those three to return
    corpus-wide values while postings() stays local, a shard produces scores
    directly comparable to every other shard -- which is the precondition for
    merging their results into one ranking.

    Nothing in BM25Ranker changes. The substitution happens entirely through
    the interface it already depends on.
    """

    def __init__(self, directory: str) -> None:
        super().__init__(directory)
        with open(Path(directory) / "index.globals", encoding="utf-8") as handle:
            globals_ = json.load(handle)
        self._global_document_count = globals_["global_document_count"]
        self._global_avgdl = globals_["global_avgdl"]
        self._global_df = globals_["global_df"]

    @property
    def document_count(self) -> int:
        return self._global_document_count

    @property
    def avgdl(self) -> float:
        return self._global_avgdl

    def document_frequency(self, term: str) -> int:
        return self._global_df.get(term, 0)

    @property
    def local_document_count(self) -> int:
        return len(self._documents)

    def local_document_frequency(self, term: str) -> int:
        """The shard-local value, kept for the divergence benchmark."""
        entry = self._terms.get(term)
        return entry[2] if entry else 0
