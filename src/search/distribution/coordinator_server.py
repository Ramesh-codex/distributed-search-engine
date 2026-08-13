from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from search.distribution.coordinator import Coordinator

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


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


class BreakerStatus(BaseModel):
    state: str
    consecutive_failures: int


class StatsResponse(BaseModel):
    shard_urls: list[str]
    breakers: dict[str, BreakerStatus]


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

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _INDEX_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request) -> StatsResponse:
        coord: Coordinator = request.app.state.coordinator
        return StatsResponse(
            shard_urls=coord.shard_urls,
            breakers={url: BreakerStatus(**info) for url, info in coord.breaker_stats.items()},
        )

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
