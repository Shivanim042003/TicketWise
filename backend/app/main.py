
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.dynamic_batcher import (
    BatchRequest,
    DynamicBatcher,
    QueueFullError,
)
from backend.app.search import search_tickets
from backend.app.semantic_search import SemanticSearch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared search resources and clean up on shutdown."""
    search_engine = SemanticSearch()
    app.state.semantic_search = search_engine

    def process_batch(
        requests: list[BatchRequest],
    ) -> list[list[dict]]:
        """Encode queries together and apply each request's top_k."""
        if not requests:
            return []

        batch_results = search_engine.search_batch(
            [item.query for item in requests],
            top_k=20,
        )

        return [
            results[:item.top_k]
            for item, results in zip(requests, batch_results)
        ]

    batcher = DynamicBatcher(
        batch_fn=process_batch,
        max_batch_size=int(os.environ.get("TICKETWISE_MAX_BATCH_SIZE", "32")),
        max_wait_ms=float(os.environ.get("TICKETWISE_MAX_WAIT_MS", "0")),
        max_queue_size=int(os.environ.get("TICKETWISE_MAX_QUEUE_SIZE", "256")),
        reject_when_full=True,
    )

    app.state.semantic_batcher = batcher
    await batcher.start()

    try:
        yield
    finally:
        await batcher.close()


app = FastAPI(
    title="TicketWise API",
    version="0.2.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


def apply_similarity_threshold(results: list[dict]) -> list[dict]:
    """Reject semantic results when the best score is below the threshold."""
    threshold = float(
        os.getenv("TICKETWISE_MIN_SIMILARITY", "0.4396")
    )

    if not results or results[0]["score"] < threshold:
        return []

    return results


@app.get("/")
def root():
    return {"message": "Welcome to TicketWise API"}


@app.get("/health")
def health_check(request: Request):
    """Report health and the backend loaded by this application."""
    search_engine: SemanticSearch = request.app.state.semantic_search

    return {
        "status": "healthy",
        "model_backend": search_engine.backend,
        "batching": request.app.state.semantic_batcher.config,
    }


@app.post("/search")
def keyword_search(request: SearchRequest):
    """Search tickets using keyword matching."""
    results = search_tickets(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "method": "keyword",
        "count": len(results),
        "results": results,
    }


@app.post("/search/semantic")
async def semantic_search(
    request: SearchRequest,
    http_request: Request,
):
    """Search tickets using dynamically batched semantic inference."""
    batcher: DynamicBatcher = (
        http_request.app.state.semantic_batcher
    )

    try:
        results = await batcher.submit(
            BatchRequest(
                query=request.query,
                top_k=request.top_k,
            )
        )
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Server is overloaded; retry shortly",
            headers={"Retry-After": "1"},
        ) from None
    results = apply_similarity_threshold(results)

    return {
        "query": request.query,
        "method": "embedding",
        "count": len(results),
        "results": results,
    }


@app.post("/search/semantic/unbatched")
def semantic_search_unbatched(
    request: SearchRequest,
    http_request: Request,
):
    """Search tickets without dynamic query batching for comparison."""
    search_engine: SemanticSearch = (
        http_request.app.state.semantic_search
    )

    results = search_engine.search(
        query=request.query,
        top_k=request.top_k,
    )
    results = apply_similarity_threshold(results)

    return {
        "query": request.query,
        "method": "embedding",
        "count": len(results),
        "results": results,
    }


@app.get("/metrics/batching")
def batching_metrics(request: Request):
    """Return runtime dynamic-batching metrics."""
    batcher: DynamicBatcher = request.app.state.semantic_batcher
    return {
        **batcher.metrics,
        "rejected_requests": batcher.rejected_requests,
    }
