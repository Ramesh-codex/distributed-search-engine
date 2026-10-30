"""Build reproducible three-shard BM25 indexes from attributed documentation excerpts.

Index format, hash partitioning, and ranking are the *existing* search engine.
The docs corpus is snapshotted to keep deployment independent of remote downloads.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from search.distribution.consistent_hash import ConsistentHashRing
from search.index.inverted_index import InvertedIndex
from search.storage.ondisk import OnDiskIndexWriter

DATA_DIR = Path(__file__).parent / "data"
SOURCES = ("fastapi", "httpx", "docker")
HOSTS = {"fastapi.tiangolo.com", "www.python-httpx.org", "docs.docker.com"}


def build_docs_index(out: Path, data_dir: Path = DATA_DIR) -> dict:
    records = []
    provenance = []
    for source in SOURCES:
        payload = json.loads((data_dir / f"{source}.json").read_text(encoding="utf-8"))
        if not payload["docs"] or not payload["sources"]:
            raise ValueError(f"Missing {source} documentation")
        records.extend(payload["docs"])
        provenance.extend(payload["sources"])

    if len(records) < 60:
        raise ValueError("Insufficient documentation corpus")
    shard_names = [f"shard-{i}" for i in range(3)]
    ring = ConsistentHashRing(shard_names)
    indexes = {name: InvertedIndex() for name in shard_names}
    metadata = {}
    source_counts = {name: 0 for name in ("FastAPI", "HTTPX", "Docker")}

    for doc_id, doc in enumerate(records, 1):
        title, body, source = doc["title"].strip(), doc["text"].strip(), doc["source"]
        link = urlsplit(doc["url"])
        if not title or len(body) < 80 or source not in source_counts:
            raise ValueError(f"Invalid documentation record #{doc_id}")
        if link.scheme != "https" or link.hostname not in HOSTS:
            raise ValueError(f"Untrusted source URL in record #{doc_id}")
        shard = ring.get_shard(f"doc:{doc_id}")
        indexes[shard].add_document(
            doc_id, f"{title} {title} {source} {body}", title=title,
            url=doc["url"]
        )
        metadata[str(doc_id)] = {
            "title": title, "section": doc["section"], "source": source,
            "url": doc["url"], "snippet": doc["summary"].strip(),
            "source_path": doc["source_path"],
        }
        source_counts[source] += 1

    if any(index.document_count == 0 for index in indexes.values()):
        raise ValueError("One or more shards are empty")

    total = len(records)
    avgdl = sum(index._total_length for index in indexes.values()) / total
    global_df = {}
    for index in indexes.values():
        for term, postings in index._postings.items():
            global_df[term] = global_df.get(term, 0) + len(postings)

    out.mkdir(parents=True, exist_ok=True)
    shard_counts = {}
    for name, index in indexes.items():
        folder = out / name
        OnDiskIndexWriter(str(folder)).write(index)
        (folder / "index.globals").write_text(json.dumps({
            "global_document_count": total, "global_avgdl": avgdl,
            "global_df": {term: global_df[term] for term in index._postings},
        }), encoding="utf-8")
        shard_counts[name] = index.document_count

    (out / "cluster.json").write_text(json.dumps({
        "shards": shard_names, "global_document_count": total,
        "global_avgdl": avgdl,
    }), encoding="utf-8")
    (out / "documents.json").write_text(json.dumps({
        "document_count": total, "source_counts": source_counts,
        "docs": metadata, "source_pages": len(provenance),
        "snapshot_date": "2026-10-30",
    }), encoding="utf-8")
    return {"document_count": total, "source_counts": source_counts,
            "source_pages": len(provenance), "shards": shard_counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs-index"))
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(build_docs_index(args.out, args.data), indent=2))
