
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"
BATCHED = "/search/semantic"
UNBATCHED = "/search/semantic/unbatched"
METRICS = "/metrics/batching"
QUERY = "API returns 503 after deployment"
WARMUP_REQUESTS = 5


def percentile(values: list[float], p: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + fraction * (
        ordered[upper] - ordered[lower]
    )


async def get_batch_metrics(client: httpx.AsyncClient) -> dict:
    """Fetch cumulative batching metrics from the API."""
    response = await client.get(METRICS)
    response.raise_for_status()
    return response.json()


def batch_metric_delta(before: dict, after: dict) -> dict:
    """Calculate metrics attributable to requests between two snapshots."""
    batches = (
        after["total_batches"] - before["total_batches"]
    )
    requests_processed = (
        after["total_requests_processed"]
        - before["total_requests_processed"]
    )

    return {
        "total_batches": batches,
        "total_requests_processed": requests_processed,
        "average_batch_size": round(
            requests_processed / batches, 2
        ) if batches else 0.0,
        # The server exposes a lifetime maximum, not a per-window maximum.
        "server_lifetime_max_observed_batch_size": (
            after["max_observed_batch_size"]
        ),
    }


async def warm_up(
    client: httpx.AsyncClient,
    endpoint: str,
) -> None:
    """Warm up an endpoint before measuring it."""
    for _ in range(WARMUP_REQUESTS):
        response = await client.post(
            endpoint,
            json={"query": QUERY, "top_k": 3},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Warm-up failed for {endpoint}: "
                f"{response.status_code} {response.text}"
            )


async def benchmark_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    requests: int,
    concurrency: int,
) -> dict:
    """Measure latency, throughput, and errors."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def send_one():
        nonlocal errors

        async with semaphore:
            start = time.perf_counter()

            try:
                response = await client.post(
                    endpoint,
                    json={"query": QUERY, "top_k": 3},
                )

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                if response.status_code == 200:
                    latencies.append(elapsed)
                else:
                    errors += 1

            except httpx.HTTPError:
                errors += 1

    start_all = time.perf_counter()

    await asyncio.gather(
        *(send_one() for _ in range(requests))
    )

    duration = time.perf_counter() - start_all
    successful = len(latencies)

    return {
        "endpoint": endpoint,
        "requests": requests,
        "concurrency": concurrency,
        "duration_s": round(duration, 4),
        "successful_requests": successful,
        "errors": errors,
        "error_rate_pct": round(
            errors / requests * 100, 2
        ),
        "throughput_rps": round(
            successful / duration, 2
        ),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_ms": round(
            statistics.mean(latencies), 2
        ) if latencies else 0.0,
    }


async def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare batched and unbatched semantic search "
            "with per-trial batching measurements."
        )
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=200,
        help="Measured requests per endpoint.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=[1, 5, 10, 20],
        default=5,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Odd trials run unbatched first; even trials run batched first.",
    )
    parser.add_argument(
        "--output",
        default="data/dynamic_batching_trials_v2.jsonl",
        help="JSONL output file.",
    )

    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")

    if args.trial < 1:
        parser.error("--trial must be at least 1")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.trial % 2 == 0:
        endpoints = [BATCHED, UNBATCHED]
    else:
        endpoints = [UNBATCHED, BATCHED]

    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=timeout,
    ) as client:
        health = await client.get("/health")
        health.raise_for_status()

        for endpoint in endpoints:
            print(f"\nBenchmarking {endpoint}...")

            await warm_up(client, endpoint)

            # Snapshot after warm-up, before measured requests.
            metrics_before = None
            if endpoint == BATCHED:
                metrics_before = await get_batch_metrics(client)

            result = await benchmark_endpoint(
                client=client,
                endpoint=endpoint,
                requests=args.requests,
                concurrency=args.concurrency,
            )

            result["trial"] = args.trial

            if endpoint == BATCHED:
                metrics_after = await get_batch_metrics(client)
                result["batch_metrics"] = batch_metric_delta(
                    metrics_before,
                    metrics_after,
                )

            print(json.dumps(result, indent=2))

            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")

    print(f"\nResults appended to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
