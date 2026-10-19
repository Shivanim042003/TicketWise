
import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmark_results.csv"
)
TEST_QUERY = "API returns 503 after deployment"



import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmark_results.csv"
)

TEST_QUERY = "API returns 503 after deployment"


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


async def benchmark_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    requests: int,
    concurrency: int,
) -> dict:
    """Benchmark an endpoint and collect successful request latencies."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    errors = 0

    async def send_request() -> None:
        nonlocal errors

        async with semaphore:
            start = time.perf_counter()

            try:
                response = await client.post(
                    endpoint,
                    json={
                        "query": TEST_QUERY,
                        "top_k": 3,
                    },
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                if response.status_code == 200:
                    latencies_ms.append(elapsed_ms)
                else:
                    errors += 1

            except httpx.HTTPError:
                errors += 1

    # Warm up the endpoint before measuring it.
    for _ in range(3):
        try:
            response = await client.post(
                endpoint,
                json={
                    "query": TEST_QUERY,
                    "top_k": 3,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Warm-up failed for {endpoint}: "
                    f"HTTP {response.status_code}"
                )

        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach {BASE_URL}{endpoint}. "
                "Is Uvicorn running?"
            ) from exc

    # Measure the full batch of concurrent requests.
    start_all = time.perf_counter()

    await asyncio.gather(
        *(send_request() for _ in range(requests))
    )

    duration_s = time.perf_counter() - start_all
    successful = len(latencies_ms)

    return {
        "endpoint": endpoint,
        "requests": requests,
        "concurrency": concurrency,
        "duration_s": round(duration_s, 4),
        "successful_requests": successful,
        "errors": errors,
        "error_rate_pct": round(
            errors / requests * 100, 2
        ),
        "throughput_rps": round(
            successful / duration_s, 2
        ),
        "p50_ms": round(
            percentile(latencies_ms, 0.50), 2
        ),
        "p95_ms": round(
            percentile(latencies_ms, 0.95), 2
        ),
        "p99_ms": round(
            percentile(latencies_ms, 0.99), 2
        ),
        "mean_ms": round(
            statistics.mean(latencies_ms), 2
        ) if latencies_ms else 0.0,
    }


async def main() -> None:
    """Parse arguments, run benchmarks, and save results."""
    parser = argparse.ArgumentParser(
        description="Benchmark TicketWise search endpoints."
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of measured requests per endpoint.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV output path for this benchmark run.",
    )

    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error(
            "--requests and --concurrency must be positive"
        )

    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=timeout,
    ) as client:
        results = []

        for endpoint in ["/search", "/search/semantic"]:
            print(f"\nBenchmarking {endpoint}...")

            result = await benchmark_endpoint(
                client=client,
                endpoint=endpoint,
                requests=args.requests,
                concurrency=args.concurrency,
            )

            results.append(result)
            print(json.dumps(result, indent=2))

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(results[0].keys())

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
