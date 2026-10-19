from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.app.search import search_tickets

app = FastAPI(
    title="TicketWise API",
    version="0.1.0",
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
def search(request: SearchRequest):
    results = search_tickets(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "count": len(results),
        "results": results,
    }
