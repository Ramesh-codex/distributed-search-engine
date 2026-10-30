"""Build deterministic read-only three-shard demo indexes at Docker build time.

This curated technical glossary is original example content, NOT the 30K-article
Wikipedia benchmark corpus. The production BM25, codec and shard classes are reused.
"""
import argparse
import json
from pathlib import Path

from search.distribution.consistent_hash import ConsistentHashRing
from search.index.inverted_index import InvertedIndex
from search.storage.ondisk import OnDiskIndexWriter


def build_demo(corpus_path: Path, output_dir: Path) -> dict:
    documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or len(documents) < 3:
        raise ValueError("demo corpus must contain at least three documents")

    shard_names = [f"shard-{i}" for i in range(3)]
    ring = ConsistentHashRing(shard_names)
    indexes = {name: InvertedIndex() for name in shard_names}

    for doc_id, doc in enumerate(documents, 1):
        title, description = doc["title"].strip(), doc["text"].strip()
        if not title or not description:
            raise ValueError(f"empty title or text for document {doc_id}")
        shard_name = ring.get_shard(f"doc:{doc_id}")
        indexes[shard_name].add_document(
            doc_id, f"{title}. {description}", title=title
        )

    if any(not index.document_count for index in indexes.values()):
        raise ValueError("demo corpus did not populate every shard")

    doc_count = sum(index.document_count for index in indexes.values())
    total_length = sum(index._total_length for index in indexes.values())
    avgdl = total_length / doc_count

    global_df = {}
    for index in indexes.values():
        for term, postings in index._postings.items():
            global_df[term] = global_df.get(term, 0) + len(postings)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for shard_name, index in indexes.items():
        folder = output_dir / shard_name
        stats = OnDiskIndexWriter(str(folder)).write(index)
        # Keep only global frequencies for terms present locally.
        globals_payload = {
            "global_document_count": doc_count,
            "global_avgdl": avgdl,
            "global_df": {term: global_df[term] for term in index._postings},
        }
        (folder / "index.globals").write_text(
            json.dumps(globals_payload), encoding="utf-8"
        )
        summary[shard_name] = {
            "documents": index.document_count,
            "terms": stats["terms"],
        }

    (output_dir / "cluster.json").write_text(
        json.dumps({
            "shards": shard_names,
            "global_document_count": doc_count,
            "global_avgdl": avgdl,
        }),
        encoding="utf-8",
    )
    return {"documents": doc_count, "shards": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path,
        default=Path(__file__).with_name("corpus.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("demo-index"))
    args = parser.parse_args()
    print(json.dumps(build_demo(args.corpus, args.out), indent=2))
