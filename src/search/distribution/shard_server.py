import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from search.distribution.shard_index import ShardIndex
from search.ranking.bm25 import BM25Ranker


class ShardResult(BaseModel):
    doc_id: int
    score: float
    title: str


class ShardStatsResponse(BaseModel):
    local_document_count: int
    global_document_count: int
    vocabulary_size: int
    avgdl: float


def create_app(shard: ShardIndex | None = None) -> FastAPI:
    """`shard=None` opens the on-disk shard at SHARD_DIR at startup -- the
    production path, one shard per container. Passing a pre-opened
    ShardIndex directly is the seam tests use to avoid touching disk.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if shard is not None:
            idx = shard
        else:
            shard_dir = os.environ.get("SHARD_DIR")
            if not shard_dir:
                raise RuntimeError("SHARD_DIR environment variable is required")
            idx = ShardIndex(shard_dir)

        app.state.shard = idx
        app.state.ranker = BM25Ranker(idx)
        yield
        idx.close()

    app = FastAPI(title="shard-server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats", response_model=ShardStatsResponse)
    def stats(request: Request) -> ShardStatsResponse:
        idx: ShardIndex = request.app.state.shard
        return ShardStatsResponse(
            local_document_count=idx.local_document_count,
            global_document_count=idx.document_count,
            vocabulary_size=idx.vocabulary_size,
            avgdl=idx.avgdl,
        )

    @app.get("/search", response_model=list[ShardResult])
    def search(
        request: Request,
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=100),
    ) -> list[ShardResult]:
        idx: ShardIndex = request.app.state.shard
        ranker: BM25Ranker = request.app.state.ranker

        results = []
        for doc_id, score in ranker.search(q, top_k=k):
            doc = idx.document(doc_id)
            results.append(ShardResult(doc_id=doc_id, score=score, title=doc.title))
        return results

    return app


app = create_app()
