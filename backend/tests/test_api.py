from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.semantic_search import SemanticSearch


def test_root():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


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
    assert len(data["results"]) <= 3
    assert data["results"][0]["id"] == "TICK-001"


def test_semantic_search():
    with TestClient(app) as client:
        response = client.post(
            "/search/semantic",
            json={
                "query": "API returns 503 after deployment",
                "top_k": 5,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "embedding"
    assert len(data["results"]) <= 5
    assert data["results"][0]["id"] == "TICK-001"
    assert any(
        ticket["id"] == "TICK-004"
        for ticket in data["results"]
    )


def test_search_rejects_empty_query():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={
                "query": "",
                "top_k": 3,
            },
        )

    assert response.status_code == 422


def test_semantic_search_rejects_invalid_backend():
    try:
        SemanticSearch(backend="onnxx")
    except ValueError as exc:
        assert "Unsupported model backend" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for unsupported backend"
        )
