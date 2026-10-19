
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from backend.app.search import search_tickets
from backend.app.semantic_search import SemanticSearch


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and compute ticket embeddings once at startup.
    app.state.semantic_search = SemanticSearch()
    yield
    # Add cleanup here if the model later owns external resources.


app = FastAPI(
    title="TicketWise API",
    version="0.2.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root():
    return {"message": "Welcome to TicketWise API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/search")
def keyword_search(request: SearchRequest):
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
def semantic_search(request: SearchRequest, http_request: Request):
    search_engine: SemanticSearch = (
        http_request.app.state.semantic_search
    )

    results = search_engine.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "method": "embedding",
        "count": len(results),
        "results": results,
    }


