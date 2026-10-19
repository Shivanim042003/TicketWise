
import json
import statistics
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("data/dynamic_batching_trials_v2.jsonl")
OUTPUT_FILE = Path("data/dynamic_batching_summary.json")


def load_results(path: Path) -> list[dict]:
    """Load benchmark results from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {path}"
        )

    results = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

    return results


def average(values: list[float]) -> float:
    """Calculate a rounded arithmetic mean."""
    return round(statistics.mean(values), 2) if values else 0.0


def summarize_endpoint(results: list[dict]) -> dict:
    """Summarize repeated measurements for one endpoint."""
    return {
        "trial_count": len(results),
        "successful_requests": sum(
            item["successful_requests"] for item in results
        ),
        "errors": sum(item["errors"] for item in results),
        "mean_throughput_rps": average(
            [item["throughput_rps"] for item in results]
        ),
        "mean_p50_ms": average(
            [item["p50_ms"] for item in results]
        ),
        "mean_p95_ms": average(
            [item["p95_ms"] for item in results]
        ),
        "mean_p99_ms": average(
            [item["p99_ms"] for item in results]
        ),
        "mean_latency_ms": average(
            [item["mean_ms"] for item in results]
        ),
    }


def main():
    results = load_results(INPUT_FILE)

    if not results:
        raise ValueError("No benchmark results were found.")

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:
        concurrency = result["concurrency"]
        endpoint = result["endpoint"]
        grouped[concurrency][endpoint].append(result)

    summary = {
        "input_file": str(INPUT_FILE),
        "methodology": {
            "requests_per_endpoint_per_trial": 200,
            "trials_per_concurrency": 2,
            "latency_summary": (
                "Arithmetic mean of trial-level percentiles; "
                "not pooled request-level percentiles."
            ),
        },
        "concurrency_results": {},
    }

    for concurrency in sorted(grouped):
        endpoints = grouped[concurrency]

        batched_results = endpoints.get(
            "/search/semantic", []
        )
        unbatched_results = endpoints.get(
            "/search/semantic/unbatched", []
        )

        batched = summarize_endpoint(batched_results)
        unbatched = summarize_endpoint(unbatched_results)

        throughput_speedup = (
            batched["mean_throughput_rps"]
            / unbatched["mean_throughput_rps"]
            if unbatched["mean_throughput_rps"]
            else 0.0
        )

        p95_reduction_pct = (
            (
                unbatched["mean_p95_ms"]
                - batched["mean_p95_ms"]
            )
            / unbatched["mean_p95_ms"]
            * 100
            if unbatched["mean_p95_ms"]
            else 0.0
        )

        batch_metrics = [
            result["batch_metrics"]
            for result in batched_results
            if "batch_metrics" in result
        ]

        observed_batch_sizes = [
            metric["average_batch_size"]
            for metric in batch_metrics
        ]

        summary["concurrency_results"][str(concurrency)] = {
            "unbatched": unbatched,
            "batched": batched,
            "throughput_speedup": round(
                throughput_speedup, 2
            ),
            "p95_latency_reduction_pct": round(
                p95_reduction_pct, 2
            ),
            "average_batch_size_by_trial": observed_batch_sizes,
            "mean_observed_batch_size": average(
                observed_batch_sizes
            ),
            "batch_counts_by_trial": [
                metric["total_batches"]
                for metric in batch_metrics
            ],
        }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Summary saved to {OUTPUT_FILE}\n")

    for concurrency, data in summary[
        "concurrency_results"
    ].items():
        print(f"Concurrency {concurrency}")
        print(
            "  Unbatched throughput: "
            f"{data['unbatched']['mean_throughput_rps']} req/s"
        )
        print(
            "  Batched throughput:   "
            f"{data['batched']['mean_throughput_rps']} req/s"
        )
        print(
            "  Throughput speedup:    "
            f"{data['throughput_speedup']}x"
        )
        print(
            "  Unbatched p95:         "
            f"{data['unbatched']['mean_p95_ms']} ms"
        )
        print(
            "  Batched p95:           "
            f"{data['batched']['mean_p95_ms']} ms"
        )
        print(
            "  p95 reduction:         "
            f"{data['p95_latency_reduction_pct']}%"
        )
        print(
            "  Mean batch size:       "
            f"{data['mean_observed_batch_size']}"
        )
        print()


if __name__ == "__main__":
    main()
