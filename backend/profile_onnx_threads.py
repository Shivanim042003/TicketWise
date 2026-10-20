import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentence_transformers
from sentence_transformers import SentenceTransformer

from backend.app.semantic_search import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data" / "evaluation_queries.json"
OUTPUT_DIR = ROOT / "data" / "profiling"

# Repeat the two most promising configurations.
THREAD_COUNTS = (4, 8)
BATCH_SIZES = (1, 5, 10, 25)
REPEATS = 100
WARMUPS = 10
ONNX_FILE = "onnx/model.onnx"


def load_queries():
    with QUERIES_PATH.open(encoding="utf-8") as file:
        queries = json.load(file)

    texts = [item["query"] for item in queries]
    if not texts:
        raise ValueError("No evaluation queries found.")

    return texts


def make_batches(texts, batch_size):
    return [
        texts[index:index + batch_size]
        for index in range(0, len(texts), batch_size)
    ]


def benchmark(model, texts, batch_size):
    batches = make_batches(texts, batch_size)

    # Warm up using the same batch shapes as the measured workload.
    for _ in range(WARMUPS):
        for batch in batches:
            model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    timings_ms = []
    measured_queries = 0

    for _ in range(REPEATS):
        for batch in batches:
            start = time.perf_counter()

            model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)
            measured_queries += len(batch)

    total_seconds = sum(timings_ms) / 1000

    # Batch sizes can be smaller than requested for the final batch.
    actual_queries_per_batch = [
        len(batch) for _ in range(REPEATS) for batch in batches
    ]

    return {
        "requested_batch_size": batch_size,
        "batches_measured": len(timings_ms),
        "queries_measured": measured_queries,
        "mean_batch_latency_ms": float(statistics.mean(timings_ms)),
        "p50_batch_latency_ms": float(np.percentile(timings_ms, 50)),
        "p95_batch_latency_ms": float(np.percentile(timings_ms, 95)),
        "p99_batch_latency_ms": float(np.percentile(timings_ms, 99)),
        "mean_per_query_latency_ms": float(
            statistics.mean(
                latency / query_count
                for latency, query_count in zip(
                    timings_ms, actual_queries_per_batch
                )
            )
        ),
        "throughput_queries_per_second": (
            measured_queries / total_seconds if total_seconds else 0.0
        ),
    }


def main():
    texts = load_queries()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "onnx_file": ONNX_FILE,
            "provider": "CPUExecutionProvider",
            "python_version": platform.python_version(),
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "onnxruntime_version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "query_count": len(texts),
            "thread_counts": list(THREAD_COUNTS),
            "batch_sizes": list(BATCH_SIZES),
            "warmup_iterations": WARMUPS,
            "repeats": REPEATS,
            "timing_scope": (
                "SentenceTransformer.encode only. Excludes HTTP, "
                "queue waiting, ranking, and model initialization."
            ),
        },
        "results": {},
    }

    for threads in THREAD_COUNTS:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1

        print(f"\nLoading ONNX model with {threads} intra-op threads...")

        model = SentenceTransformer(
            MODEL_NAME,
            backend="onnx",
            model_kwargs={
                "provider": "CPUExecutionProvider",
                "file_name": ONNX_FILE,
                "session_options": options,
            },
        )

        results["results"][str(threads)] = {}

        for batch_size in BATCH_SIZES:
            print(
                f"Measuring threads={threads}, "
                f"batch_size={batch_size}..."
            )

            measurement = benchmark(model, texts, batch_size)
            results["results"][str(threads)][str(batch_size)] = measurement

            print(
                f"  mean={measurement['mean_batch_latency_ms']:.3f} ms, "
                f"p50={measurement['p50_batch_latency_ms']:.3f} ms, "
                f"p95={measurement['p95_batch_latency_ms']:.3f} ms, "
                f"p99={measurement['p99_batch_latency_ms']:.3f} ms, "
                f"throughput="
                f"{measurement['throughput_queries_per_second']:.2f} q/s"
            )

        del model

    output_path = OUTPUT_DIR / "onnx_thread_comparison_repeat.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSummary")
    print("=" * 88)
    print(
        f"{'Threads':>8}{'Batch':>8}{'Mean ms':>12}"
        f"{'p95 ms':>12}{'p99 ms':>12}{'Queries/s':>14}"
    )

    for threads in THREAD_COUNTS:
        for batch_size in BATCH_SIZES:
            item = results["results"][str(threads)][str(batch_size)]
            print(
                f"{threads:>8}{batch_size:>8}"
                f"{item['mean_batch_latency_ms']:>12.3f}"
                f"{item['p95_batch_latency_ms']:>12.3f}"
                f"{item['p99_batch_latency_ms']:>12.3f}"
                f"{item['throughput_queries_per_second']:>14.2f}"
            )

    print(f"\nSaved repeat results to: {output_path}")


if __name__ == "__main__":
    main()
