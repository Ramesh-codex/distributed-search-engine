import heapq
import math

from search.index.inverted_index import InvertedIndex
from search.index.tokenizer import tokenize

# k1 controls term-frequency saturation: how fast the benefit of repeated
# occurrences flattens out. b controls length normalisation: 0 ignores
# document length entirely, 1 normalises fully. Standard empirical defaults.
K1 = 1.2
B = 0.75


class BM25Ranker:
    def __init__(self, index: InvertedIndex, k1: float = K1, b: float = B) -> None:
        self._index = index
        self._k1 = k1
        self._b = b

    def _idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF with +0.5 smoothing.

        The +1 inside the log keeps this non-negative: without it a term in
        more than half the corpus scores negative and penalises documents
        that contain it.
        """
        n = self._index.document_count
        df = self._index.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Score documents containing any query term, return top K.

        Only documents in a postings list are ever touched -- the entire point
        of an inverted index. Selection uses a heap: O(n log k), not O(n log n).
        """
        avgdl = self._index.avgdl
        if avgdl == 0:
            return []

        scores: dict[int, float] = {}

        for term in tokenize(query):
            idf = self._idf(term)
            if idf == 0.0:
                continue

            for doc_id, tf in self._index.postings(term):
                doc = self._index.document(doc_id)
                if doc is None:
                    continue

                # tf appears in numerator and denominator, so the score
                # approaches idf*(k1+1) asymptotically rather than growing
                # without bound. Keyword stuffing stops paying.
                norm = 1 - self._b + self._b * (doc.length / avgdl)
                weight = (tf * (self._k1 + 1)) / (tf + self._k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * weight

        return heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
