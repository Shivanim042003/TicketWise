import pytest
from fastapi.testclient import TestClient

from backend.app.dynamic_batcher import DynamicBatcher, QueueFullError
from backend.app.main import app
from backend.app.semantic_search import SemanticSearch


def test_root():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to TicketWise API"}


def test_health(monkeypatch):
    for name in (
        "TICKETWISE_MODEL_BACKEND",
        "TICKETWISE_MAX_BATCH_SIZE",
        "TICKETWISE_MAX_WAIT_MS",
        "TICKETWISE_MAX_QUEUE_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_backend"] == "pytorch"
    assert body["batching"] == {
        "max_batch_size": 32,
        "max_wait_ms": 0.0,
        "max_queue_size": 256,
    }


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
    payloads = [
        {"query": "API returns 503 after deployment", "top_k": 3},
        {"query": "database connection timeout", "top_k": 3},
        {"query": "login fails after password reset", "top_k": 3},
    ]

    with TestClient(app) as client:
        before = client.get("/metrics/batching").json()

        for payload in payloads:
            response = client.post("/search/semantic", json=payload)
            assert response.status_code == 200

        after = client.get("/metrics/batching").json()

    assert before["total_batches"] == 0
    assert before["total_requests_processed"] == 0
    assert before["average_batch_size"] == 0.0
    assert before["rejected_requests"] == 0

    assert after["total_requests_processed"] == len(payloads)
    assert 1 <= after["total_batches"] <= len(payloads)
    assert after["average_batch_size"] == pytest.approx(
        len(payloads) / after["total_batches"],
        abs=0.01,
    )
    assert after["max_observed_batch_size"] >= after["average_batch_size"]
    assert after["rejected_requests"] == 0


def test_semantic_search_returns_503_when_queue_full(monkeypatch):
    with TestClient(app) as client:

        async def full(item):
            raise QueueFullError("full")

        monkeypatch.setattr(app.state.semantic_batcher, "submit", full)
        response = client.post("/search/semantic", json={"query": "anything"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


def test_batcher_rejects_when_queue_full_if_configured():
    import asyncio
    import threading

    release = threading.Event()

    def slow(items):
        release.wait(5)
        return items

    async def scenario():
        batcher = DynamicBatcher(
            slow,
            max_batch_size=1,
            max_wait_ms=0,
            max_queue_size=1,
            reject_when_full=True,
        )
        await batcher.start()

        first = asyncio.create_task(batcher.submit("a"))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(batcher.submit("b"))
        await asyncio.sleep(0.1)

        with pytest.raises(QueueFullError):
            await batcher.submit("c")

        assert batcher.rejected_requests == 1

        release.set()
        assert await first == "a"
        assert await second == "b"
        await batcher.close()

    asyncio.run(scenario())


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
