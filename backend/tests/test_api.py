import pytest
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
    with pytest.raises(
        ValueError,
        match="Unsupported model backend",
    ):
        SemanticSearch(backend="onnxx")


def test_concurrent_semantic_search_respects_each_top_k():
    from concurrent.futures import ThreadPoolExecutor

    requests = [
        {
            "query": "API returns 503 after deployment",
            "top_k": 1,
        },
        {
            "query": "database connection timeout",
            "top_k": 3,
        },
        {
            "query": "login fails after password reset",
            "top_k": 2,
        },
    ]

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(
                executor.map(
                    lambda payload: client.post(
                        "/search/semantic",
                        json=payload,
                    ),
                    requests,
                )
            )

    for response, payload in zip(responses, requests):
        assert response.status_code == 200

        body = response.json()

        assert body["query"] == payload["query"]
        assert body["count"] == payload["top_k"]
        assert len(body["results"]) == payload["top_k"]


def test_batching_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics/batching")

    assert response.status_code == 200

    data = response.json()

    expected_fields = {
        "total_batches",
        "total_requests_processed",
        "average_batch_size",
        "max_observed_batch_size",
    }

    assert expected_fields.issubset(data.keys())

    assert data["total_batches"] >= 0
    assert data["total_requests_processed"] >= 0
    assert data["average_batch_size"] >= 0
    assert data["max_observed_batch_size"] >= 0

    if data["total_batches"] == 0:
        assert data["total_requests_processed"] == 0
        assert data["average_batch_size"] == 0.0
    else:
        assert data["total_requests_processed"] > 0

        assert data["average_batch_size"] == pytest.approx(
            data["total_requests_processed"]
            / data["total_batches"]
        )

        assert (
            data["max_observed_batch_size"]
            >= data["average_batch_size"]
        )
def test_similarity_threshold_rejects_unrelated_query():
    with TestClient(app) as client:
        response = client.post(
            "/search/semantic",
            json={"query": "What is the weather forecast tomorrow?"},
        )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["count"] == 0


def test_similarity_threshold_keeps_relevant_query():
    with TestClient(app) as client:
        response = client.post(
            "/search/semantic",
            json={"query": "API returns 503 after deployment"},
        )

    assert response.status_code == 200
    assert response.json()["count"] > 0
    assert response.json()["results"][0]["id"] == "TICK-001"
