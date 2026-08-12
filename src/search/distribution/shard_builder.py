import json
from pathlib import Path

from search.distribution.consistent_hash import ConsistentHashRing
from search.index.inverted_index import InvertedIndex
from search.storage.ondisk import OnDiskIndexWriter
from search.storage.wiki_loader import stream_articles


def build_shards(
    dump: str,
    out_dir: str,
    shard_count: int = 3,
    limit: int = 100000,
) -> dict:
    """Partition a corpus across N shards, each with global BM25 statistics.

    Why global stats are written into every shard: BM25's IDF term depends on
    corpus-wide N and document frequency, and avgdl on corpus-wide mean length.
    A shard computing those from its own slice produces scores that are not
    comparable to another shard's, so merging them ranks arbitrarily. Because
    this index is built offline and immutable, the global values are free to
    compute here and cost nothing at query time.

    Documents are routed by consistent hash rather than round-robin so that
    adding a fourth shard later reassigns roughly a quarter of documents
    instead of reshuffling all of them.
    """
    shard_names = [f"shard-{i}" for i in range(shard_count)]
    ring = ConsistentHashRing(shard_names)
    indexes = {name: InvertedIndex() for name in shard_names}

    for doc_id, (title, text) in enumerate(stream_articles(dump, limit=limit), 1):
        target = ring.get_shard(f"doc:{doc_id}")
        indexes[target].add_document(doc_id, text, title=title)

    total_docs = sum(ix.document_count for ix in indexes.values())
    total_length = sum(ix._total_length for ix in indexes.values())
    global_avgdl = total_length / total_docs if total_docs else 0.0

    # Global df: a term's document frequency is the sum across every shard.
    global_df: dict[str, int] = {}
    for ix in indexes.values():
        for term, postings in ix._postings.items():
            global_df[term] = global_df.get(term, 0) + len(postings)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, ix in indexes.items():
        shard_dir = out / name
        stats = OnDiskIndexWriter(str(shard_dir)).write(ix)

        # Only the terms this shard actually holds; shipping all 564K to
        # every shard would triple the dictionary for no benefit.
        local_global_df = {t: global_df[t] for t in ix._postings}
        with open(shard_dir / "index.globals", "w", encoding="utf-8") as handle:
            json.dump({
                "global_document_count": total_docs,
                "global_avgdl": global_avgdl,
                "global_df": local_global_df,
            }, handle)

        summary[name] = {
            "documents": ix.document_count,
            "vocabulary": ix.vocabulary_size,
            "postings_bytes": stats["postings_bytes"],
        }

    with open(out / "cluster.json", "w", encoding="utf-8") as handle:
        json.dump({
            "shards": shard_names,
            "global_document_count": total_docs,
            "global_avgdl": global_avgdl,
        }, handle)

    return {
        "total_documents": total_docs,
        "global_avgdl": round(global_avgdl, 2),
        "global_vocabulary": len(global_df),
        "shards": summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="data/dumps/simplewiki.xml.bz2")
    parser.add_argument("--out", default="data/cluster")
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--size", type=int, default=100000)
    args = parser.parse_args()

    result = build_shards(args.dump, args.out, args.shards, args.size)
    print(json.dumps(result, indent=2))
