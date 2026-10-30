# Developer Documentation Search (new application on the original search engine)

A working real-documentation search vertical built on the existing custom
inverted index, BM25 ranker, consistent-hash partitioning, on-disk postings,
FastAPI shard servers and scatter/gather HTTP coordinator.

## What is indexed

188 short sections from 20 actual documentation pages:
FastAPI (10 pages), HTTPX (6 pages), and Docker (4 pages), captured
2026-10-30 from official open-source repositories. These are excerpt snapshots,
**not** complete documentation websites. Each hit links to the official
documentation page and shows the original source name, a small excerpt, the
responding shard, and a BM25 score.

Snapshots are reproducible and committed to this branch in
`docs_app/data/*.json`. The source blob hashes and original URLs are
recorded beside each excerpt. Licenses are preserved in
`docs_app/licenses/`: FastAPI MIT; HTTPX BSD-3-Clause; Docker docs Apache-2.0.
This is a technical portfolio demo, not an authoritative replacement for
current official documentation.

## Run local HTTP system

```bash
python -m pip install -r requirements-demo.txt
export PYTHONPATH=src:.
python -m docs_app.build_index --out docs-index
DEMO_INDEX_DIR="$(pwd)/docs-index" DOCS_METADATA_FILE="$(pwd)/docs-index/documents.json" PORT=10000 python -m docs_app.run_cluster
```

Then visit http://localhost:10000 and search, for example,
`dependency injection`, `request timeout`, or `Dockerfile`.
JSON API: `/search?q=request%20timeout&source=HTTPX&k=5`;
statistics: `/stats`; docs: `/docs`.

Run the self-contained smoke test (it builds a temporary index and starts the
real four-process HTTP system):

```bash
PYTHONPATH=src:. python -m docs_app.smoke_test
```

Run through Docker:

```bash
docker build -f Dockerfile.docs -t developer-docs-search .
docker run --rm -p 10000:10000 developer-docs-search
```

## Optional Render deployment

Use `render.docs.yaml` as a separate Blueprint, selecting the
`developer-docs-search` branch. It proposes **one new free** Docker web
service named `developer-docs-search-demo`, not a modification to your
already-live `distributed-search-portfolio-demo`. Before deploying, check
Render's total workspace free-instance-hour allowance; two services could
use that allowance faster. Do not choose a paid plan by accident.

## Design decisions and current limitations

- Prebuilt immutable indexes keep startup deterministic and do not require
  downloading a documentation website at container start.
- The shard processes are separate HTTP processes but share a single host
  for minimum cost; this is not independent machine-level redundancy.
- Source filters are applied after bounded top-100 distributed retrieval;
  large-corpus production search would require shard-side filtering/indexing.
- No document update endpoint; refresh the snapshot and rebuild the image
  when new documentation is available.
- The results count reflects returned hits, not all corpus-wide matches.
- No semantic embeddings, managed database, ingestion queue or LLM yet.
- The original Wikipedia benchmark results remain untouched and do not
  describe the latency of this smaller documentation dataset.
