import gc
import json
import random
import statistics
import subprocess
import sys
import time

import psutil

from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
CORPUS_SIZES = (5000, 10000, 20000)


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_one(limit: int) -> dict:
    """Measured in a dedicated process.

    CPython does not return freed arenas to the OS promptly, so measuring
    several corpus sizes in one process makes later indexes appear to cost
    less than earlier ones -- they are being built inside memory the
    allocator already holds. A cold process is the only honest baseline.
    """
    gc.collect()
    baseline = rss_mb()

    index = InvertedIndex()
    titles = []
    start = time.perf_counter()
    for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=limit), 1):
        index.add_document(doc_id, text, title=title)
        titles.append(title)
    build_seconds = time.perf_counter() - start

    peak = rss_mb() - baseline

    ranker = BM25Ranker(index)
    rng = random.Random(42)
    queries = [rng.choice(titles) for _ in range(300)]

    for q in queries[:20]:          # warm-up
        ranker.search(q)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        ranker.search(q, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()

    return {
        "docs": index.document_count,
        "build_seconds": round(build_seconds, 1),
        "docs_per_sec": round(index.document_count / build_seconds),
        "vocab": index.vocabulary_size,
        "avgdl": round(index.avgdl),
        "index_mb": round(peak, 1),
        "kb_per_doc": round(peak * 1024 / index.document_count, 1),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
    }


if __name__ == "__main__":
    # Child mode: one corpus size, result as JSON on stdout.
    if len(sys.argv) == 2:
        print(json.dumps(run_one(int(sys.argv[1]))))
        sys.exit(0)

    rows = []
    for size in CORPUS_SIZES:
        print(f"measuring {size} documents in a fresh process...", flush=True)
        completed = subprocess.run(
            [sys.executable, __file__, str(size)],
            capture_output=True, text=True, check=True,
        )
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    columns = ["docs", "build_seconds", "docs_per_sec", "vocab", "avgdl",
               "index_mb", "kb_per_doc", "p50_ms", "p95_ms", "p99_ms"]

    print()
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in columns) + " |")
