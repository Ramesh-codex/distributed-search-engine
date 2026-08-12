import json
import mmap
import struct
from pathlib import Path

from search.index.codec import decode_postings, encode_postings
from search.index.inverted_index import Document, InvertedIndex


class OnDiskIndexWriter:
    """Serialises an in-memory index to a memory-mappable on-disk format.

    Postings go into one contiguous blob rather than one bytes object per
    term. At 221K terms that saves 221K object headers -- roughly 7 MB of
    pure overhead, and more importantly it means the reader holds a single
    mapping instead of a dict full of small allocations.
    """

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, index: InvertedIndex) -> dict:
        terms: dict[str, tuple[int, int, int]] = {}
        offset = 0

        with open(self._dir / "index.postings", "wb") as blob:
            for term in sorted(index._postings):
                postings = index._postings[term]
                encoded = encode_postings(postings)
                blob.write(encoded)
                terms[term] = (offset, len(encoded), len(postings))
                offset += len(encoded)

        with open(self._dir / "index.terms", "w", encoding="utf-8") as handle:
            json.dump(terms, handle)

        docs = {
            str(doc_id): [d.url, d.title, d.length]
            for doc_id, d in index._documents.items()
        }
        with open(self._dir / "index.docs", "w", encoding="utf-8") as handle:
            json.dump({"documents": docs, "total_length": index._total_length}, handle)

        return {
            "postings_bytes": offset,
            "terms": len(terms),
            "documents": len(docs),
        }


class OnDiskIndex:
    """Read-only index backed by a memory-mapped postings blob.

    The mapping is lazy: the OS pages in only the slices actually touched by
    queries, so a postings file larger than available RAM still works. This
    is what removes the in-memory ceiling -- the previous implementation had
    to hold every posting resident before answering a single query.
    """

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)

        with open(self._dir / "index.terms", encoding="utf-8") as handle:
            self._terms: dict[str, list[int]] = json.load(handle)

        with open(self._dir / "index.docs", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._documents = {
            int(doc_id): Document(int(doc_id), url, title, length)
            for doc_id, (url, title, length) in payload["documents"].items()
        }
        self._total_length = payload["total_length"]

        self._file = open(self._dir / "index.postings", "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def postings(self, term: str) -> list[tuple[int, int]]:
        entry = self._terms.get(term)
        if entry is None:
            return []
        offset, length, _ = entry
        return decode_postings(self._map[offset:offset + length])

    def document_frequency(self, term: str) -> int:
        """Read from the term table, not by decoding: df is needed for every
        query term to compute IDF, and decoding a long postings list just to
        count it would defeat the point of storing the length."""
        entry = self._terms.get(term)
        return entry[2] if entry else 0

    def document(self, doc_id: int) -> Document | None:
        return self._documents.get(doc_id)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._terms)

    @property
    def avgdl(self) -> float:
        if not self._documents:
            return 0.0
        return self._total_length / len(self._documents)

    def close(self) -> None:
        self._map.close()
        self._file.close()
