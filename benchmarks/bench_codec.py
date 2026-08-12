import gc
import sys

import psutil

from search.index.codec import encode_postings
from search.index.inverted_index import InvertedIndex
from search.storage.wiki_loader import stream_articles

DUMP = "data/dumps/simplewiki.xml.bz2"
LIMIT = 20000


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


gc.collect()
baseline = rss_mb()

index = InvertedIndex()
for doc_id, (title, text) in enumerate(stream_articles(DUMP, limit=LIMIT), 1):
    index.add_document(doc_id, text, title=title)

tuple_mb = rss_mb() - baseline
print(f"docs        {index.document_count}")
print(f"vocab       {index.vocabulary_size}")
print(f"tuple index {tuple_mb:8.1f} MB   {tuple_mb*1024/index.document_count:6.1f} KB/doc")

# Encode every postings list and measure the packed bytes directly. sys.getsizeof
# includes the bytes object header (~33 B), which is the honest comparison --
# a real index still pays one object header per term.
packed_bytes = 0
raw_postings = 0
for term in list(index._postings):
    postings = index._postings[term]
    raw_postings += len(postings)
    packed_bytes += sys.getsizeof(encode_postings(postings))

packed_mb = packed_bytes / (1024 * 1024)
print(f"postings    {raw_postings:,} entries")
print(f"packed      {packed_mb:8.1f} MB   {packed_mb*1024/index.document_count:6.1f} KB/doc")
print(f"bytes/posting  packed {packed_bytes/raw_postings:.2f}")
