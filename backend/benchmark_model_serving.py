
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from backend.app.semantic_search import MODEL_NAME
from backend.app.search import load_tickets


ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data" / "evaluation_queries.json"
OUTPUT_DIR = ROOT / "data" / "profiling"

WARMUP_ITERATIONS = 5
REPEATS_PER_QUERY = 100
BATCH_SIZES = (1, 5, 10, 25)
TOP_K_VALUES = (1, 3, 5)


def percentile(values, p):
    """Calculate a percentile using NumPy."""
    return float(np.percentile(values, p))


def load_json(path):
    """Load JSON data from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(ranked_ids, relevant_ids, k):
    """Calculate recall at k for a query."""
    if not relevant_ids:
        return 0.0

    return (
        len(set(ranked_ids[:k]) & relevant_ids)
        / len(relevant_ids)
    )


def reciprocal_rank(ranked_ids, relevant_ids):
    """Calculate reciprocal rank for a query."""
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def rank_tickets(model, tickets, queries):
    """Evaluate retrieval quality using normalized embeddings."""
    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embeddings = model.encode(
        [item["query"] for item in queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    scores_matrix = query_embeddings @ document_embeddings.T
    rankings = []

    for item, scores in zip(queries, scores_matrix):
        indices = scores.argsort()[::-1]
        ranked_ids = [tickets[i]["id"] for i in indices]
        relevant_ids = set(item["relevant_ticket_ids"])

        rankings.append(
            {
                "query": item["query"],
                "ranked_ids": ranked_ids,
                "relevant_ids": sorted(relevant_ids),
                "recall": {
                    str(k): recall_at_k(
                        ranked_ids,
                        relevant_ids,
                        k,
                    )
                    for k in TOP_K_VALUES
                },
                "reciprocal_rank": reciprocal_rank(
                    ranked_ids,
                    relevant_ids,
                ),
            }
        )

    metrics = {
        f"recall_at_{k}": float(
            np.mean(
                [
                    item["recall"][str(k)]
                    for item in rankings
                ]
            )
        )
        for k in TOP_K_VALUES
    }

    metrics["mrr"] = float(
        np.mean(
            [
                item["reciprocal_rank"]
                for item in rankings
            ]
        )
    )

    return rankings, metrics


def benchmark_batch_encoding(model, query_texts, batch_size):
    """Measure model encoding latency and throughput."""
    timings_ms = []
    total_queries = len(query_texts)

    # Warm up the model using the same batch size.
    warmup_batch = query_texts[:batch_size]

    for _ in range(WARMUP_ITERATIONS):
        model.encode(
            warmup_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # Include a potentially smaller final batch.
    batches = [
        query_texts[index:index + batch_size]
        for index in range(0, total_queries, batch_size)
    ]
    batches = [batch for batch in batches if batch]

    # Retain the actual size of every measured batch so that
    # per-query latency is calculated correctly.
    measured_batch_sizes = []

    for _ in range(REPEATS_PER_QUERY):
        for batch in batches:
            start = time.perf_counter()

            model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            timings_ms.append(elapsed_ms)
            measured_batch_sizes.append(len(batch))

    measured_queries = (
        sum(len(batch) for batch in batches)
        * REPEATS_PER_QUERY
    )

    total_seconds = sum(timings_ms) / 1000

    mean_per_query_latency_ms = statistics.mean(
        latency / actual_batch_size
        for latency, actual_batch_size in zip(
            timings_ms,
            measured_batch_sizes,
        )
    )

    return {
        "batch_size": batch_size,
        "batches_measured": len(timings_ms),
        "queries_measured": measured_queries,
        "mean_batch_latency_ms": float(
            statistics.mean(timings_ms)
        ),
        "p50_batch_latency_ms": percentile(
            timings_ms,
            50,
        ),
        "p95_batch_latency_ms": percentile(
            timings_ms,
            95,
        ),
        "p99_batch_latency_ms": percentile(
            timings_ms,
            99,
        ),
        "throughput_queries_per_second": (
            measured_queries / total_seconds
            if total_seconds
            else 0.0
        ),
        "mean_per_query_latency_ms": float(
            mean_per_query_latency_ms
        ),
    }


def main():
    """Run retrieval evaluation and model encoding benchmarks."""
    tickets = load_tickets()
    queries = load_json(QUERIES_PATH)
    query_texts = [
        item["query"]
        for item in queries
    ]

    if not tickets or not queries:
        raise ValueError(
            "Tickets and evaluation queries must be non-empty."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Tickets: {len(tickets)}")
    print(f"Evaluation queries: {len(queries)}")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(f"Repeats per query: {REPEATS_PER_QUERY}")

    results = {
        "metadata": {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model": MODEL_NAME,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "onnxruntime_version": onnxruntime.__version__,
            "onnx_providers_available": (
                onnxruntime.get_available_providers()
            ),
            "device": "CPU",
            "ticket_count": len(tickets),
            "evaluation_query_count": len(queries),
            "warmup_iterations": WARMUP_ITERATIONS,
            "repeats_per_query": REPEATS_PER_QUERY,
            "batch_sizes": list(BATCH_SIZES),
            "timing_scope": (
                "Model.encode only; excludes HTTP, queue wait, "
                "similarity calculation, ranking, and model "
                "initialization."
            ),
            "per_query_latency_method": (
                "Arithmetic mean of each measured batch's "
                "latency divided by its actual batch size."
            ),
        },
        "backends": {},
    }

    models = {}

    for name, backend in (
        ("pytorch", "torch"),
        ("onnx", "onnx"),
    ):
        print(f"\nLoading {name} model...")

        if backend == "onnx":
            model = SentenceTransformer(
                MODEL_NAME,
                backend="onnx",
                model_kwargs={
                    "provider": "CPUExecutionProvider",
                },
            )
        else:
            model = SentenceTransformer(
                MODEL_NAME,
                device="cpu",
            )

        models[name] = model

        print(f"Evaluating retrieval quality for {name}...")

        rankings, metrics = rank_tickets(
            model,
            tickets,
            queries,
        )

        results["backends"][name] = {
            "retrieval_metrics": metrics,
            "rankings": rankings,
            "latency": {},
        }

        for batch_size in BATCH_SIZES:
            print(
                f"  Benchmarking batch size {batch_size}..."
            )

            measurement = benchmark_batch_encoding(
                model,
                query_texts,
                batch_size,
            )

            results["backends"][name]["latency"][
                str(batch_size)
            ] = measurement

    pytorch_rankings = results["backends"][
        "pytorch"
    ]["rankings"]

    onnx_rankings = results["backends"][
        "onnx"
    ]["rankings"]

    results["ranking_consistency"] = {
        "identical_full_rankings": sum(
            pytorch["ranked_ids"] == onnx["ranked_ids"]
            for pytorch, onnx in zip(
                pytorch_rankings,
                onnx_rankings,
            )
        ),
        "total_queries": len(queries),
    }

    output_path = (
        OUTPUT_DIR / "model_serving_comparison.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nRetrieval quality")
    print("=" * 72)
    print(
        f"{'Backend':<12}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR':>12}"
    )

    for name, backend_results in results[
        "backends"
    ].items():
        metrics = backend_results["retrieval_metrics"]

        print(
            f"{name:<12}"
            f"{metrics['recall_at_1']:>12.3f}"
            f"{metrics['recall_at_3']:>12.3f}"
            f"{metrics['recall_at_5']:>12.3f}"
            f"{metrics['mrr']:>12.3f}"
        )

    print("\nBatch encoding latency and throughput")
    print("=" * 100)
    print(
        f"{'Backend':<12}"
        f"{'Batch':>8}"
        f"{'Mean ms':>12}"
        f"{'p50 ms':>12}"
        f"{'p95 ms':>12}"
        f"{'p99 ms':>12}"
        f"{'Mean ms/query':>17}"
        f"{'Queries/s':>14}"
    )

    for name, backend_results in results[
        "backends"
    ].items():
        for batch_size in BATCH_SIZES:
            item = backend_results["latency"][
                str(batch_size)
            ]

            print(
                f"{name:<12}"
                f"{batch_size:>8}"
                f"{item['mean_batch_latency_ms']:>12.3f}"
                f"{item['p50_batch_latency_ms']:>12.3f}"
                f"{item['p95_batch_latency_ms']:>12.3f}"
                f"{item['p99_batch_latency_ms']:>12.3f}"
                f"{item['mean_per_query_latency_ms']:>17.3f}"
                f"{item['throughput_queries_per_second']:>14.2f}"
            )

    print("\nRanking consistency")
    print("=" * 72)
    print(
        "Identical full rankings: "
        f"{results['ranking_consistency']['identical_full_rankings']}"
        f"/{results['ranking_consistency']['total_queries']}"
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
