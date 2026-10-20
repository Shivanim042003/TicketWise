import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.app.semantic_search import SemanticSearch


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERIES_PATH = "data/evaluation_queries_v2.json"
QUERIES_PATH = (
    ROOT
    / os.getenv("TICKETWISE_EVAL_DATASET", DEFAULT_QUERIES_PATH)
).resolve()

TOP_K_VALUES = (1, 3, 5)
BACKENDS = ("pytorch", "onnx")
VALID_QUERY_TYPES = {"paraphrase", "hard_negative", "no_match"}


def load_queries():
    """Load and validate the selected evaluation dataset."""
    if not QUERIES_PATH.is_relative_to(ROOT):
        raise ValueError(
            "Evaluation dataset must be located inside the project directory."
        )

    if not QUERIES_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {QUERIES_PATH}\n"
            "Set TICKETWISE_EVAL_DATASET to a valid project-relative JSON path."
        )

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list) or not queries:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"Query {index} must be a JSON object.")

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query {index} has no valid query string.")

        query_type = item.get("type")
        if query_type not in VALID_QUERY_TYPES:
            raise ValueError(
                f"Query {index} has unsupported type: {query_type!r}"
            )

        relevant_ids = item.get("relevant_ticket_ids")
        if not isinstance(relevant_ids, list):
            raise ValueError(
                f"Query {index} must contain relevant_ticket_ids as a list."
            )

        if any(
            not isinstance(ticket_id, str) or not ticket_id.strip()
            for ticket_id in relevant_ids
        ):
            raise ValueError(
                f"Query {index} contains an invalid ticket ID."
            )

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(
                f"Query {index} contains duplicate relevant ticket IDs."
            )

        if query_type == "no_match" and relevant_ids:
            raise ValueError(
                f"No-match query {index} must have an empty relevance list."
            )

        if query_type != "no_match" and not relevant_ids:
            raise ValueError(
                f"Query {index} must have at least one relevant ticket."
            )

    return queries


def recall_at_k(ranked_ids, relevant_ids, k):
    """Calculate recall at k."""
    if not relevant_ids:
        return None

    return (
        len(set(ranked_ids[:k]) & set(relevant_ids))
        / len(set(relevant_ids))
    )


def reciprocal_rank(ranked_ids, relevant_ids):
    """Calculate reciprocal rank of the first relevant result."""
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return None

    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_backend(backend, queries):
    """Evaluate retrieval ranking for one inference backend."""
    os.environ["TICKETWISE_MODEL_BACKEND"] = backend
    engine = SemanticSearch(backend=backend)

    ticket_ids = {ticket["id"] for ticket in engine.tickets}

    for index, item in enumerate(queries):
        unknown_ids = set(item["relevant_ticket_ids"]) - ticket_ids
        if unknown_ids:
            raise ValueError(
                f"Query {index} references unknown ticket IDs: "
                f"{sorted(unknown_ids)}"
            )

    grouped = defaultdict(list)
    rankings = []

    for item in queries:
        results = engine.search(item["query"], top_k=5)
        ranked_ids = [result["id"] for result in results]
        relevant_ids = item["relevant_ticket_ids"]

        row = {
            "query": item["query"],
            "type": item["type"],
            "ranked_ids": ranked_ids,
            "relevant_ticket_ids": relevant_ids,
            "recall": {
                str(k): recall_at_k(ranked_ids, relevant_ids, k)
                for k in TOP_K_VALUES
            },
            "reciprocal_rank": reciprocal_rank(
                ranked_ids,
                relevant_ids,
            ),
            "top_score": results[0]["score"] if results else None,
        }

        rankings.append(row)
        grouped[item["type"]].append(row)

    def summarize(rows):
        metrics = {}

        for k in TOP_K_VALUES:
            values = [
                row["recall"][str(k)]
                for row in rows
                if row["recall"][str(k)] is not None
            ]
            metrics[f"recall_at_{k}"] = (
                float(np.mean(values)) if values else None
            )

        mrr_values = [
            row["reciprocal_rank"]
            for row in rows
            if row["reciprocal_rank"] is not None
        ]
        metrics["mrr"] = (
            float(np.mean(mrr_values)) if mrr_values else None
        )
        metrics["query_count"] = len(rows)

        return metrics

    return {
        "backend": backend,
        "overall": summarize(rankings),
        "by_type": {
            query_type: summarize(rows)
            for query_type, rows in sorted(grouped.items())
        },
        "rankings": rankings,
    }


def main():
    queries = load_queries()

    results = {
        "evaluation_dataset": str(QUERIES_PATH.relative_to(ROOT)),
        "query_count": len(queries),
        "backends": {},
    }

    for backend in BACKENDS:
        print(f"\nEvaluating {backend}...")
        results["backends"][backend] = evaluate_backend(
            backend,
            queries,
        )

    pytorch_rows = results["backends"]["pytorch"]["rankings"]
    onnx_rows = results["backends"]["onnx"]["rankings"]

    results["ranking_consistency"] = {
        "identical_full_rankings": sum(
            pytorch["ranked_ids"] == onnx["ranked_ids"]
            for pytorch, onnx in zip(pytorch_rows, onnx_rows)
        ),
        "total_queries": len(queries),
    }

    output_path = (
        ROOT
        / "data"
        / "profiling"
        / f"retrieval_evaluation_{QUERIES_PATH.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for backend in BACKENDS:
        result = results["backends"][backend]

        print(f"\n{backend.upper()} overall:")
        print(json.dumps(result["overall"], indent=2))
        print(f"\n{backend.upper()} by query type:")
        print(json.dumps(result["by_type"], indent=2))

    print("\nRanking consistency:")
    print(json.dumps(results["ranking_consistency"], indent=2))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
