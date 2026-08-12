from collections import Counter, defaultdict
from dataclasses import dataclass

from search.index.tokenizer import tokenize


@dataclass(slots=True)
class Document:
    """Metadata the ranker needs. `length` is post-analysis token count."""
    doc_id: int
    url: str
    title: str
    length: int


class DuplicateDocumentError(Exception):
    pass


class InvertedIndex:
    """term -> [(doc_id, term_frequency), ...], postings sorted by doc_id.

    Sorted by doc_id, not frequency: it makes multi-term queries a linear
    merge-join, and it is the precondition for delta encoding later, since
    gaps between sorted ids are small and compress well. Frequency-sorted
    postings would suit WAND-style early termination -- a later optimisation
    that would need a second copy of the data.
    """

    def __init__(self) -> None:
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._documents: dict[int, Document] = {}
        self._total_length = 0

    def add_document(self, doc_id: int, text: str, url: str = "", title: str = "") -> None:
        if doc_id in self._documents:
            # Appending twice would put one doc in a postings list twice,
            # which breaks document_frequency(), which breaks IDF.
            raise DuplicateDocumentError(f"doc_id {doc_id} already indexed")

        # Counter aggregates occurrences in one pass, so each document
        # contributes exactly one entry per term -- the invariant that lets
        # document_frequency() just read len(postings).
        frequencies = Counter(tokenize(text))
        length = sum(frequencies.values())

        for term, tf in frequencies.items():
            self._postings[term].append((doc_id, tf))

        self._documents[doc_id] = Document(doc_id, url, title, length)
        self._total_length += length

    def postings(self, term: str) -> list[tuple[int, int]]:
        return self._postings.get(term, [])

    def document_frequency(self, term: str) -> int:
        """Derived, not stored: len(postings) IS df, given one entry per doc."""
        return len(self._postings.get(term, []))

    def document(self, doc_id: int) -> Document | None:
        return self._documents.get(doc_id)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def avgdl(self) -> float:
        """Maintained incrementally. Recomputing per query would turn an
        O(postings) lookup into an O(corpus) walk."""
        if not self._documents:
            return 0.0
        return self._total_length / len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)