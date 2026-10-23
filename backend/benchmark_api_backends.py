
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 60

CONCURRENCIES = (1, 5, 10)
REQUESTS_PER_RUN = 100
WARMUP_REQUESTS = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERY_FILE = PROJECT_ROOT / "data" / "evaluation_queries.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "profiling"


def percentile(values, pct):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_queries():
    """Load evaluation queries without modifying the source file."""
    with QUERY_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in ("queries", "evaluation_queries", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"No query list found in {QUERY_FILE}. "
            "Check the JSON structure."
        )

    queries = []

    for item in data:
        if isinstance(item, str):
            query = item
        elif isinstance(item, dict):
            query = (
                item.get("query")
                or item.get("text")
                or item.get("question")
            )
        else:
            query = None

        if isinstance(query, str) and query.strip():
            queries.append(query.strip())

    if not queries:
        raise ValueError("No usable queries found in evaluation_queries.json.")

    return queries


async def get_batch_metrics(client):
    """Fetch batching metrics when the endpoint is available."""
    try:
        response = await client.get("/metrics/batching")
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            return payload

    except (httpx.HTTPError, ValueError):
        pass

    return None


def metric_delta(before, after):
    """Calculate numeric counter deltas, not deltas of averages/maxima."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None

    delta = {}

    # These are counters. Averages and maxima must be interpreted
    # separately because subtracting them does not give run statistics.
    counter_keys = (
        "total_batches",
        "total_requests_processed",
    )

    for key in counter_keys:
        old_value = before.get(key)
        new_value = after.get(key)

        if (
            isinstance(old_value, (int, float))
            and isinstance(new_value, (int, float))
        ):
            delta[key] = new_value - old_value

    batches = delta.get("total_batches", 0)
    requests = delta.get("total_requests_processed", 0)

    delta["requests_per_batch"] = (
        requests / batches if batches > 0 else None
    )

    return delta


async def verify_server_backend(client, expected_backend):
    """Verify the backend reported by the running API process."""
    response = await client.get("/health")
    response.raise_for_status()

    try:
        health_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "The /health endpoint did not return valid JSON."
        ) from exc

    actual_backend = health_data.get("model_backend")

    if actual_backend != expected_backend:
        raise RuntimeError(
            f"Backend mismatch: benchmark expects {expected_backend!r}, "
            f"but the server reports {actual_backend!r}. "
            "Restart the API with the intended "
            "TICKETWISE_MODEL_BACKEND value."
        )

    print(f"Health endpoint: {response.status_code}")
    print(f"Verified server backend: {actual_backend}")


async def warmup(client, queries):
    """Warm up the API before measuring latency."""
    print(f"Running {WARMUP_REQUESTS} warm-up requests...")

    for index in range(WARMUP_REQUESTS):
        query = queries[index % len(queries)]

        try:
            response = await client.post(
                "/search/semantic",
                json={"query": query},
            )
            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Warm-up failed. Check API startup, model loading, "
                "and the semantic search request schema. "
                f"Details: {exc}"
            ) from exc


async def benchmark_load(client, queries, concurrency):
    """Measure latency and throughput at a specific concurrency."""
    latencies = []
    errors = []
    semaphore = asyncio.Semaphore(concurrency)

    async def send_one(index):
        query = queries[index % len(queries)]

        async with semaphore:
            start = time.perf_counter()

            try:
                response = await client.post(
                    "/search/semantic",
                    json={"query": query},
                )

                elapsed_ms = (time.perf_counter() - start) * 1000
                response.raise_for_status()

                # Validate the response structure, not just JSON syntax.
                payload = response.json()

                if not isinstance(payload, dict):
                    raise ValueError("Semantic response must be a JSON object.")

                results = payload.get("results")
                count = payload.get("count")

                if (
                    payload.get("method") != "embedding"
                    or not isinstance(results, list)
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count != len(results)
                ):
                    raise ValueError(
                        "Unexpected semantic search response structure."
                    )

                latencies.append(elapsed_ms)

            except (httpx.HTTPError, ValueError) as exc:
                errors.append({
                    "request_index": index,
                    "query": query,
                    "error": str(exc),
                })

    metrics_before = await get_batch_metrics(client)
    start_all = time.perf_counter()

    await asyncio.gather(
        *(send_one(index) for index in range(REQUESTS_PER_RUN))
    )

    duration = time.perf_counter() - start_all
    metrics_after = await get_batch_metrics(client)

    successful = len(latencies)
    error_count = len(errors)

    return {
        "requests": REQUESTS_PER_RUN,
        "concurrency": concurrency,
        "duration_seconds": round(duration, 6),
        "successful_requests": successful,
        "errors": error_count,
        "error_rate_pct": round(
            100 * error_count / REQUESTS_PER_RUN, 3
        ),
        "throughput_rps": round(
            successful / duration if duration else 0.0, 3
        ),
        "mean_ms": round(
            statistics.mean(latencies) if latencies else 0.0, 3
        ),
        "p50_ms": round(percentile(latencies, 50), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "p99_ms": round(percentile(latencies, 99), 3),
        "batch_metrics_before": metrics_before,
        "batch_metrics_after": metrics_after,
        "batch_counter_delta": metric_delta(
            metrics_before, metrics_after
        ),
        "error_samples": errors[:5],
    }


async def run_backend(backend, queries):
    print(f"\n{'=' * 65}")
    print(f"Benchmark backend: {backend}")
    print(f"API: {BASE_URL}")
    print(f"{'=' * 65}")

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=httpx.Timeout(TIMEOUT_SECONDS),
    ) as client:
        try:
            await verify_server_backend(client, backend)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot verify API at {BASE_URL}. "
                "Start the API server first. "
                f"Details: {exc}"
            ) from exc

        await warmup(client, queries)

        backend_results = {
            "backend": backend,
            "base_url": BASE_URL,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "requests_per_run": REQUESTS_PER_RUN,
            "warmup_requests": WARMUP_REQUESTS,
            "concurrency_levels": list(CONCURRENCIES),
            "runs": [],
        }

        for concurrency in CONCURRENCIES:
            print(
                f"\nTesting concurrency={concurrency}, "
                f"requests={REQUESTS_PER_RUN}"
            )

            result = await benchmark_load(
                client,
                queries,
                concurrency,
            )
            backend_results["runs"].append(result)

            print(
                f"  Success: {result['successful_requests']}/"
                f"{result['requests']}"
            )
            print(f"  Errors: {result['errors']}")
            print(
                f"  Throughput: {result['throughput_rps']} requests/sec"
            )
            print(
                f"  Latency mean/p50/p95/p99: "
                f"{result['mean_ms']}/"
                f"{result['p50_ms']}/"
                f"{result['p95_ms']}/"
                f"{result['p99_ms']} ms"
            )
            print(
                "  Batch counter delta:",
                result["batch_counter_delta"],
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"api_benchmark_{backend}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(backend_results, file, indent=2)

    print(f"\nSaved results to: {output_path}")

    return backend_results


def main():
    backend = os.environ.get("BENCHMARK_BACKEND", "").strip().lower()

    if backend not in {"pytorch", "onnx"}:
        raise ValueError(
            "Set BENCHMARK_BACKEND to pytorch or onnx. "
            "Example: BENCHMARK_BACKEND=onnx "
            "python -m backend.benchmark_api_backends"
        )

    queries = load_queries()
    print(f"Loaded {len(queries)} evaluation queries.")

    asyncio.run(run_backend(backend, queries))


if __name__ == "__main__":
    main()
