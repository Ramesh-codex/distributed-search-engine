"""Developer docs HTTP API, reusing the real distributed Coordinator."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from search.distribution.coordinator import Coordinator

STATIC_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
DEFAULT_METADATA = "/app/docs-index/documents.json"
ALLOWED_SOURCES = {"all", "FastAPI", "HTTPX", "Docker"}


def create_app(coordinator: Coordinator | None = None, metadata: dict | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.docs = (metadata if metadata is not None else json.loads(
            Path(os.getenv("DOCS_METADATA_FILE", DEFAULT_METADATA)).read_text(
                encoding="utf-8"
            )
        ))
        app.state.coordinator = coordinator if coordinator is not None else Coordinator()
        try:
            yield
        finally:
            app.state.coordinator.close()

    app = FastAPI(title="Developer Documentation Search", version="1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return STATIC_HTML

    @app.get("/health")
    def health(request: Request):
        if not request.app.state.docs.get("docs"):
            raise HTTPException(status_code=503, detail="Index metadata unavailable")
        return {"status": "ok"}

    @app.get("/stats")
    def stats(request: Request):
        coord = request.app.state.coordinator
        docs = request.app.state.docs
        return {"shard_urls": coord.shard_urls, "breakers": coord.breaker_stats,
                "document_count": docs["document_count"],
                "source_counts": docs["source_counts"],
                "source_pages": docs["source_pages"],
                "snapshot_date": docs["snapshot_date"]}

    @app.get("/search")
    def search(request: Request, q: str = Query(..., min_length=1, max_length=160),
               k: int = Query(10, ge=1, le=20), source: str = Query("all")):
        if source not in ALLOWED_SOURCES:
            raise HTTPException(status_code=422, detail="Unsupported documentation source")
        coord = request.app.state.coordinator
        # Fetch a wider top-K before applying a source filter.
        # This is bounded; results_count is *not* an exact corpus-wide hit total.
        raw = coord.search(q, top_k=100)
        docs = request.app.state.docs["docs"]
        enriched = []
        for hit in raw["results"]:
            doc = docs.get(str(hit["doc_id"]))
            if doc is None or (source != "all" and doc["source"] != source):
                continue
            enriched.append({
                "doc_id": hit["doc_id"], "title": doc["title"],
                "section": doc["section"], "source": doc["source"],
                "url": doc["url"], "snippet": doc["snippet"],
                "score": round(hit["score"], 4), "shard": hit["shard"],
            })
            if len(enriched) >= k:
                break
        return {"query": q, "source": source, "results": enriched,
                "elapsed_ms": raw["elapsed_ms"],
                "shards_queried": raw["shards_queried"],
                "shards_failed": raw["shards_failed"],
                "degraded": raw["degraded"]}

    return app


app = create_app()
