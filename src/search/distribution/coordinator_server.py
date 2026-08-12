from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from search.distribution.coordinator import Coordinator


class ShardResultOut(BaseModel):
    doc_id: int
    score: float
    shard: str
    title: str


class SearchResponse(BaseModel):
    query: str
    results: list[ShardResultOut]
    elapsed_ms: float
    shards_queried: int
    shards_failed: list[str]
    degraded: bool


def create_app(coordinator: Coordinator | None = None) -> FastAPI:
    """`coordinator=None` builds a Coordinator from SHARD_URLS at startup --
    the production path, one process fronting every shard. Passing a
    pre-built Coordinator directly (e.g. one wired to an httpx MockTransport)
    is the seam tests use to avoid real shard services.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.coordinator = coordinator if coordinator is not None else Coordinator()
        yield
        app.state.coordinator.close()

    app = FastAPI(title="search-coordinator", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=100),
    ) -> SearchResponse:
        coord: Coordinator = request.app.state.coordinator
        return coord.search(q, top_k=k)

    return app


app = create_app()
