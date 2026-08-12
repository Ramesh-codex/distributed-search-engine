import random
import statistics
import time

from search.index.inverted_index import InvertedIndex
from search.ranking.bm25 import BM25Ranker
from search.ranking.cached_ranker import CachedRanker
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
LIMIT = 20000
TRIALS = 2000

index = InvertedIndex()
titles = []
for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=LIMIT), 1):
    index.add_document(doc_id, text, title=title)
    titles.append(title)
print(f"indexed {index.document_count}")

# Zipfian query distribution: real search traffic is dominated by a small
# number of popular queries. A uniform random draw over 20K titles would
# show a near-zero hit rate and tell you nothing about production behaviour.
rng = random.Random(42)
popular = rng.sample(titles, 200)
weights = [1.0 / (i + 1) for i in range(len(popular))]
queries = rng.choices(popular, weights=weights, k=TRIALS)


def measure(ranker, label):
    for q in queries[:50]:
        ranker.search(q)
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        ranker.search(q, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    print(f"\n{label}")
    print(f"  p50 {statistics.median(latencies):8.3f} ms")
    print(f"  p95 {latencies[int(len(latencies)*0.95)]:8.3f} ms")
    print(f"  p99 {latencies[int(len(latencies)*0.99)]:8.3f} ms")
    print(f"  max {latencies[-1]:8.3f} ms")
    return latencies


measure(BM25Ranker(index), "uncached")

cached = CachedRanker(index, capacity=50)
measure(cached, "cached (capacity 256)")
print(f"  {cached.stats}")
