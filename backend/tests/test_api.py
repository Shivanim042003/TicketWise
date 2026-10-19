import pytest
from backend.app.semantic_search import SemanticSearch
from fastapi.testclient import TestClient

from backend.app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to TicketWise API"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_keyword_search():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={
                "query": "API returns 503 after deployment",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "keyword"
    assert data["results"][0]["id"] == "TICK-001"


def test_semantic_search():
    with TestClient(app) as client:
        response = client.post(
            "/search/semantic",
            json={
                "query": "API returns 503 after deployment",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "embedding"
    assert data["results"][0]["id"] == "TICK-001"
    assert any(
        ticket["id"] == "TICK-004"
        for ticket in data["results"]
    )


def test_search_rejects_empty_query():
    with TestClient(app) as client:
        response = client.post(
            "/search/semantic",
            json={"query": "", "top_k": 3},
        )

    assert response.status_code == 422
def test_semantic_search_rejects_invalid_backend():
    with pytest.raises(ValueError, match="Unsupported model backend"):
        SemanticSearch(backend="onnxx")
