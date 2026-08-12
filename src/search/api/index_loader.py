import logging
import os
import time
from urllib.parse import quote

from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.wiki_loader import stream_articles

log = logging.getLogger(__name__)

DEFAULT_DUMP_PATH = "data/dumps/simplewiki.xml.bz2"
DEFAULT_CORPUS_SIZE = 20000


def _article_url(title: str) -> str:
    return f"https://simple.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def build_index() -> tuple[InvertedIndex, BM25Ranker]:
    """Builds the corpus once, meant to be called a single time at process
    startup -- never per request. Corpus size and dump path are env-configurable
    so the same code serves a quick dev instance or a full run unchanged.
    """
    dump_path = os.environ.get("SEARCH_DUMP_PATH", DEFAULT_DUMP_PATH)
    corpus_size = int(os.environ.get("SEARCH_CORPUS_SIZE", DEFAULT_CORPUS_SIZE))

    log.info("building index: %d docs from %s", corpus_size, dump_path)
    start = time.perf_counter()

    index = InvertedIndex()
    for doc_id, (title, text) in enumerate(stream_articles(dump_path, limit=corpus_size), 1):
        index.add_document(doc_id, text, url=_article_url(title), title=title)

    elapsed = time.perf_counter() - start
    log.info(
        "indexed %d docs in %.1fs (%d docs/sec)",
        index.document_count, elapsed, round(index.document_count / elapsed),
    )

    return index, BM25Ranker(index)
