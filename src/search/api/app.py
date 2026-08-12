import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from search.api.index_loader import build_index
from search.index.inverted_index import InvertedIndex
from search.index.tokenizer import tokenize
from search.ranking.bm25 import BM25Ranker

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


class SearchResult(BaseModel):
    doc_id: int
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    elapsed_ms: float
    results: list[SearchResult]


class StatsResponse(BaseModel):
    document_count: int
    vocabulary_size: int
    avgdl: float


def create_app(index: InvertedIndex | None = None) -> FastAPI:
    """`index=None` builds the real corpus from the wiki dump at startup --
    the production path. Passing a pre-built index skips that and is what
    lets tests exercise the API against a small in-memory index instead of
    a multi-minute dump load.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if index is not None:
            app.state.index = index
            app.state.ranker = BM25Ranker(index)
        else:
            app.state.index, app.state.ranker = build_index()
        yield

    app = FastAPI(title="distributed-search-engine", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _INDEX_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request) -> StatsResponse:
        idx: InvertedIndex = request.app.state.index
        return StatsResponse(
            document_count=idx.document_count,
            vocabulary_size=idx.vocabulary_size,
            avgdl=idx.avgdl,
        )

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=100),
    ) -> SearchResponse:
        idx: InvertedIndex = request.app.state.index
        ranker: BM25Ranker = request.app.state.ranker

        start = time.perf_counter()
        hits = ranker.search(q, top_k=k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Distinct doc_ids matching >=1 query term: the true hit count,
        # independent of the top_k truncation applied to `hits`.
        matched_ids = {doc_id for term in tokenize(q) for doc_id, _ in idx.postings(term)}

        results = []
        for doc_id, score in hits:
            doc = idx.document(doc_id)
            results.append(SearchResult(doc_id=doc_id, title=doc.title, url=doc.url, score=round(score, 4)))

        return SearchResponse(
            query=q,
            total_hits=len(matched_ids),
            elapsed_ms=round(elapsed_ms, 2),
            results=results,
        )

    return app


app = create_app()
