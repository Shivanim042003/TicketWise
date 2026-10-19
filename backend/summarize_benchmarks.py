
import csv
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BACKENDS = {
    "PyTorch": [
        DATA_DIR / f"controlled_pytorch_{i}.csv"
        for i in range(1, 4)
    ],
    "ONNX": [
        DATA_DIR / f"controlled_onnx_{i}.csv"
        for i in range(1, 4)
    ],
}

METRICS = [
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "throughput_rps",
    "error_rate_pct",
]


def load_results(paths):
    results = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark file: {path}"
            )

        with path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            for row in csv.DictReader(file):
                row["trial"] = path.stem
                results.append(row)

    return results


def main():
    summaries = {}

    for backend, paths in BACKENDS.items():
        rows = load_results(paths)

        for endpoint in ["/search", "/search/semantic"]:
            selected = [
                row for row in rows
                if row["endpoint"] == endpoint
            ]

            if len(selected) != 3:
                raise ValueError(
                    f"Expected 3 trials for {backend} "
                    f"{endpoint}, found {len(selected)}"
                )

            print(f"\n{backend} — {endpoint}")
            print("-" * 55)

            for metric in METRICS:
                values = [
                    float(row[metric])
                    for row in selected
                ]
                median = statistics.median(values)
                spread = max(values) - min(values)

                print(
                    f"{metric:18} "
                    f"median={median:8.2f}  "
                    f"range={min(values):.2f}–"
                    f"{max(values):.2f}  "
                    f"spread={spread:.2f}"
                )

                summaries[
                    (backend, endpoint, metric)
                ] = median

            errors = sum(
                int(row["errors"]) for row in selected
            )
            requests = sum(
                int(row["requests"]) for row in selected
            )

            print(f"Total requests: {requests}")
            print(f"Total errors:   {errors}")

    print("\nONNX comparison against PyTorch")
    print("=" * 55)

    for endpoint in ["/search", "/search/semantic"]:
        print(f"\n{endpoint}")

        for metric in METRICS:
            pytorch = summaries[
                ("PyTorch", endpoint, metric)
            ]
            onnx = summaries[
                ("ONNX", endpoint, metric)
            ]

            if pytorch == 0:
                continue

            if metric == "error_rate_pct":
                difference = onnx - pytorch
                print(
                    f"{metric:18} "
                    f"PyTorch={pytorch:.2f}, "
                    f"ONNX={onnx:.2f}, "
                    f"difference={difference:+.2f} "
                    "percentage points"
                )
                continue

            change_pct = (onnx - pytorch) / pytorch * 100

            interpretation = (
                "lower is better"
                if metric in {
                    "p50_ms", "p95_ms", "p99_ms", "mean_ms"
                }
                else "higher is better"
            )

            print(
                f"{metric:18} "
                f"PyTorch={pytorch:.2f}, "
                f"ONNX={onnx:.2f}, "
                f"change={change_pct:+.1f}% "
                f"({interpretation})"
            )


if __name__ == "__main__":
    main()
