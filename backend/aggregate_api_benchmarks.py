
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "profiling"

BACKENDS = ("onnx", "pytorch")
TRIALS = (1, 2, 3)
CONCURRENCIES = (1, 5, 10)

LATENCY_METRICS = (
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
)


def load_trial(backend, trial):
    path = DATA_DIR / (
        f"api_benchmark_{backend}_trial{trial}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing trial file: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("backend") != backend:
        raise ValueError(
            f"{path} has backend={data.get('backend')!r}; "
            f"expected {backend!r}."
        )

    return data


def validate_trial(data, backend, trial):
    """Check that the file contains the expected benchmark runs."""
    runs = data.get("runs", [])
    found = {
        run.get("concurrency"): run
        for run in runs
    }

    missing = set(CONCURRENCIES) - set(found)

    if missing:
        raise ValueError(
            f"{backend} trial {trial} is missing concurrency "
            f"levels: {sorted(missing)}"
        )

    for concurrency in CONCURRENCIES:
        run = found[concurrency]

        if run.get("requests") != 100:
            raise ValueError(
                f"{backend} trial {trial}, concurrency "
                f"{concurrency}: expected 100 requests."
            )

        if run.get("successful_requests", 0) + run.get("errors", 0) != 100:
            raise ValueError(
                f"{backend} trial {trial}, concurrency "
                f"{concurrency}: request counts do not total 100."
            )

    return found


def summarize(values):
    """Return median, minimum, maximum, and range."""
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for backend in BACKENDS:
        loaded[backend] = {}

        for trial in TRIALS:
            data = load_trial(backend, trial)
            loaded[backend][trial] = validate_trial(
                data, backend, trial
            )

    summary_rows = []

    for concurrency in CONCURRENCIES:
        row = {"concurrency": concurrency}

        for backend in BACKENDS:
            runs = [
                loaded[backend][trial][concurrency]
                for trial in TRIALS
            ]

            for metric in ("throughput_rps",) + LATENCY_METRICS:
                values = [float(run[metric]) for run in runs]
                stats = summarize(values)

                prefix = f"{backend}_{metric}"

                for stat_name, stat_value in stats.items():
                    row[f"{prefix}_{stat_name}"] = round(
                        stat_value, 3
                    )

            row[f"{backend}_total_requests"] = sum(
                run["requests"] for run in runs
            )
            row[f"{backend}_successful_requests"] = sum(
                run["successful_requests"] for run in runs
            )
            row[f"{backend}_errors"] = sum(
                run["errors"] for run in runs
            )

            batch_counts = []
            requests_per_batch = []

            for run in runs:
                batch_delta = run.get("batch_counter_delta") or {}
                if "total_batches" in batch_delta:
                    batch_counts.append(
                        batch_delta["total_batches"]
                    )
                if batch_delta.get("requests_per_batch") is not None:
                    requests_per_batch.append(
                        batch_delta["requests_per_batch"]
                    )

            if len(batch_counts) == len(TRIALS):
                row[f"{backend}_batches_median"] = round(
                    statistics.median(batch_counts), 3
                )

            if len(requests_per_batch) == len(TRIALS):
                row[f"{backend}_requests_per_batch_median"] = round(
                    statistics.median(requests_per_batch), 3
                )

        onnx_tput = row["onnx_throughput_rps_median"]
        pytorch_tput = row["pytorch_throughput_rps_median"]

        row["onnx_throughput_improvement_pct"] = round(
            (onnx_tput / pytorch_tput - 1) * 100
            if pytorch_tput else 0.0,
            2,
        )

        onnx_p95 = row["onnx_p95_ms_median"]
        pytorch_p95 = row["pytorch_p95_ms_median"]

        row["onnx_p95_latency_reduction_pct"] = round(
            (1 - onnx_p95 / pytorch_p95) * 100
            if pytorch_p95 else 0.0,
            2,
        )

        summary_rows.append(row)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials_per_backend": len(TRIALS),
        "requests_per_concurrency_per_trial": 100,
        "concurrency_levels": list(CONCURRENCIES),
        "method": (
            "Median, minimum, maximum and range across three trials. "
            "Latency and throughput are summarized separately."
        ),
        "results": summary_rows,
    }

    json_path = DATA_DIR / "api_benchmark_aggregate.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    csv_path = DATA_DIR / "api_benchmark_aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = DATA_DIR / "api_benchmark_report.md"

    lines = [
        "# TicketWise API Backend Benchmark",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "Three trials per backend, 100 requests per concurrency "
        "level per trial.",
        "",
        "Medians are reported across the three trials. "
        "Throughput improvement is relative to PyTorch. "
        "Positive p95 latency reduction means ONNX is faster.",
        "",
        "| Concurrency | Backend | Median throughput (req/s) "
        "| Range (req/s) | Median p50 (ms) | Median p95 (ms) "
        "| Median p99 (ms) | Errors / 300 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        for backend in BACKENDS:
            lines.append(
                f"| {row['concurrency']} "
                f"| {backend.upper()} "
                f"| {row[f'{backend}_throughput_rps_median']:.2f} "
                f"| {row[f'{backend}_throughput_rps_range']:.2f} "
                f"| {row[f'{backend}_p50_ms_median']:.2f} "
                f"| {row[f'{backend}_p95_ms_median']:.2f} "
                f"| {row[f'{backend}_p99_ms_median']:.2f} "
                f"| {row[f'{backend}_errors']} / "
                f"{row[f'{backend}_total_requests']} |"
            )

    lines.extend([
        "",
        "## ONNX vs PyTorch",
        "",
        "| Concurrency | Throughput improvement | "
        "p95 latency reduction |",
        "|---:|---:|---:|",
    ])

    for row in summary_rows:
        lines.append(
            f"| {row['concurrency']} "
            f"| {row['onnx_throughput_improvement_pct']:+.2f}% "
            f"| {row['onnx_p95_latency_reduction_pct']:+.2f}% |"
        )

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- Range is maximum minus minimum across three trials.",
        "- Three trials provide preliminary evidence, not a "
        "statistical guarantee.",
        "- These are local benchmark results; they do not directly "
        "predict production performance.",
        "- Batching averages are summarized only when all three "
        "trials report them.",
        "",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nAggregation complete.")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print(f"Markdown:    {md_path}")

    print("\nMedian throughput comparison:")
    print("Concurrency | ONNX req/s | PyTorch req/s | ONNX improvement")

    for row in summary_rows:
        print(
            f"{row['concurrency']:>11} | "
            f"{row['onnx_throughput_rps_median']:>10.2f} | "
            f"{row['pytorch_throughput_rps_median']:>13.2f} | "
            f"{row['onnx_throughput_improvement_pct']:>+9.2f}%"
        )


if __name__ == "__main__":
    main()
