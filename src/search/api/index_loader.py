import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

from search.index.inverted_index import InvertedIndex
from search.storage.ondisk import OnDiskIndex, OnDiskIndexWriter
from search.storage.wiki_loader import stream_articles

log = logging.getLogger(__name__)

DEFAULT_DUMP_PATH = "data/dumps/simplewiki.xml.bz2"
DEFAULT_CORPUS_SIZE = 20000

_ONDISK_FILES = ("index.postings", "index.terms", "index.docs")


def article_url(title: str) -> str:
    return f"https://simple.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def _has_prebuilt_index(index_dir: str) -> bool:
    directory = Path(index_dir)
    return all((directory / name).exists() for name in _ONDISK_FILES)


def build_in_memory_index(dump_path: str, corpus_size: int) -> InvertedIndex:
    log.info("building index: %d docs from %s", corpus_size, dump_path)
    start = time.perf_counter()

    index = InvertedIndex()
    for doc_id, (title, text) in enumerate(stream_articles(dump_path, limit=corpus_size), 1):
        index.add_document(doc_id, text, url=article_url(title), title=title)

    elapsed = time.perf_counter() - start
    log.info(
        "indexed %d docs in %.1fs (%d docs/sec)",
        index.document_count, elapsed, round(index.document_count / elapsed),
    )
    return index


def build_index() -> InvertedIndex | OnDiskIndex:
    """Called once at process startup -- never per request.

    If SEARCH_INDEX_DIR points at a directory already holding a prebuilt
    on-disk index, load it via mmap and skip the dump entirely. Otherwise
    build in memory from the dump as before, then -- if SEARCH_INDEX_DIR is
    set -- persist that index so the next startup can take the fast path.
    """
    index_dir = os.environ.get("SEARCH_INDEX_DIR")
    if index_dir and _has_prebuilt_index(index_dir):
        log.info("loading prebuilt on-disk index from %s", index_dir)
        return OnDiskIndex(index_dir)

    dump_path = os.environ.get("SEARCH_DUMP_PATH", DEFAULT_DUMP_PATH)
    corpus_size = int(os.environ.get("SEARCH_CORPUS_SIZE", DEFAULT_CORPUS_SIZE))
    index = build_in_memory_index(dump_path, corpus_size)

    if index_dir:
        log.info("writing on-disk index to %s for next startup", index_dir)
        OnDiskIndexWriter(index_dir).write(index)

    return index
