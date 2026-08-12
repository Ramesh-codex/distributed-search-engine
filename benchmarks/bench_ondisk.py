import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.storage.ondisk import OnDiskIndex, OnDiskIndexWriter
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
INDEX_DIR = "data/index100k"
LIMIT = 100000
TRIALS = 300


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def sample_titles():
    return [t for t, _ in zip(
        (title for title, _ in stream_articles(DUMP, limit=200)), range(200))]


def latency(index, queries):
    ranker = BM25Ranker(index)
    for q in queries[:20]:
        ranker.search(q)
    out = []
    for q in queries:
        t0 = time.perf_counter()
        ranker.search(q, top_k=10)
        out.append((time.perf_counter() - t0) * 1000)
    out.sort()
    return {
        "p50_ms": round(statistics.median(out), 3),
        "p95_ms": round(out[int(len(out) * 0.95)], 3),
        "p99_ms": round(out[int(len(out) * 0.99)], 3),
    }


def run(mode: str) -> dict:
    """Each mode runs in a cold process so RSS is not polluted by the other."""
    queries = sample_titles()
    gc.collect()
    baseline = rss_mb()

    if mode == "memory":
        index = InvertedIndex()
        start = time.perf_counter()
        for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=LIMIT), 1):
            index.add_document(doc_id, text, title=title)
        load_s = time.perf_counter() - start
        result = {"resident_mb": round(rss_mb() - baseline, 1),
                  "load_seconds": round(load_s, 1)}
        result.update(latency(index, queries))
        return result

    start = time.perf_counter()
    index = OnDiskIndex(INDEX_DIR)
    load_s = time.perf_counter() - start
    result = {"resident_mb": round(rss_mb() - baseline, 1),
              "load_seconds": round(load_s, 2)}
    result.update(latency(index, queries))
    index.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(json.dumps(run(sys.argv[1])))
        sys.exit(0)

    if not Path(INDEX_DIR, "index.postings").exists():
        print("building on-disk index...", flush=True)
        ix = InvertedIndex()
        for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=LIMIT), 1):
            ix.add_document(doc_id, text, title=title)
        stats = OnDiskIndexWriter(INDEX_DIR).write(ix)
        print(stats, flush=True)
        del ix
        gc.collect()

    on_disk = sum(f.stat().st_size for f in Path(INDEX_DIR).iterdir()) / (1024 * 1024)
    print(f"on-disk footprint {on_disk:.1f} MB\n", flush=True)

    rows = {}
    for mode in ("memory", "ondisk"):
        print(f"measuring {mode} in a fresh process...", flush=True)
        done = subprocess.run([sys.executable, __file__, mode],
                              capture_output=True, text=True, check=True)
        rows[mode] = json.loads(done.stdout.strip().splitlines()[-1])

    cols = ["resident_mb", "load_seconds", "p50_ms", "p95_ms", "p99_ms"]
    print("\n| mode | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for mode, row in rows.items():
        print(f"| {mode} | " + " | ".join(str(row[c]) for c in cols) + " |")
