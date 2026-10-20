import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation_queries.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = (1, 3, 5)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of labeled relevant tickets present in the top k."""
    if not relevant_ids:
        return 0.0

    retrieved_ids = set(ranked_ids[:k])
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def first_relevant_rank(ranked_ids, relevant_ids):
    """Return the rank of the first relevant ticket, or None."""
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return rank

    return None


def evaluate_backend(name, model, tickets, queries):
    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    # Encode documents once per backend.
    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_results = []

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ticket_ids"])

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Normalized embeddings make dot product equal cosine similarity.
        scores = document_embeddings @ query_embedding
        indices = scores.argsort()[::-1]

        ranked_ids = [tickets[i]["id"] for i in indices]
        ranked_titles = [tickets[i]["title"] for i in indices]

        query_results.append(
            {
                "query": query,
                "relevant_ids": sorted(relevant_ids),
                "ranked_ids": ranked_ids,
                "ranked_titles": ranked_titles,
                "first_relevant_rank": first_relevant_rank(
                    ranked_ids, relevant_ids
                ),
                "recall": {
                    k: recall_at_k(ranked_ids, relevant_ids, k)
                    for k in K_VALUES
                },
                "reciprocal_rank": reciprocal_rank(
                    ranked_ids, relevant_ids
                ),
            }
        )

    if not query_results:
        raise ValueError("No evaluation queries found.")

    return {
        "name": name,
        "query_results": query_results,
        "recall": {
            k: sum(result["recall"][k] for result in query_results)
            / len(query_results)
            for k in K_VALUES
        },
        "mrr": sum(
            result["reciprocal_rank"] for result in query_results
        ) / len(query_results),
    }


def main():
    tickets = load_json(TICKETS_PATH)
    queries = load_json(QUERIES_PATH)

    if not tickets:
        raise ValueError("No tickets found.")

    if not queries:
        raise ValueError("No evaluation queries found.")

    ticket_ids = {ticket["id"] for ticket in tickets}

    for item in queries:
        unknown_ids = set(item["relevant_ticket_ids"]) - ticket_ids
        if unknown_ids:
            raise ValueError(
                f"Query {item['query']!r} references unknown tickets: "
                f"{sorted(unknown_ids)}"
            )

    print(f"Tickets: {len(tickets)}")
    print(f"Evaluation queries: {len(queries)}")

    print("\nLoading PyTorch model...")
    pytorch_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Loading ONNX model...")
    onnx_model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider"},
    )

    results = [
        evaluate_backend("PyTorch", pytorch_model, tickets, queries),
        evaluate_backend("ONNX", onnx_model, tickets, queries),
    ]

    print("\nAggregate retrieval comparison")
    print("=" * 78)
    print(
        f"{'Backend':<12}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR':>12}"
    )
    print("-" * 78)

    for result in results:
        print(
            f"{result['name']:<12}"
            f"{result['recall'][1]:>12.3f}"
            f"{result['recall'][3]:>12.3f}"
            f"{result['recall'][5]:>12.3f}"
            f"{result['mrr']:>12.3f}"
        )

    pytorch_results = results[0]["query_results"]
    onnx_results = results[1]["query_results"]

    matching_queries = sum(
        pytorch["ranked_ids"] == onnx["ranked_ids"]
        for pytorch, onnx in zip(pytorch_results, onnx_results)
    )

    print(
        f"\nIdentical full rankings: "
        f"{matching_queries}/{len(queries)} queries"
    )

    for result in results:
        print(f"\n{result['name']}: per-query diagnostics")
        print("-" * 78)

        for item in result["query_results"]:
            rank = item["first_relevant_rank"]
            rank_text = str(rank) if rank is not None else "not found"

            print(f"\nQuery: {item['query']}")
            print(f"  Relevant IDs: {item['relevant_ids']}")
            print(f"  Top 5 IDs:    {item['ranked_ids'][:5]}")
            print(f"  First relevant rank: {rank_text}")
            print(
                "  Recall@1/3/5: "
                f"{item['recall'][1]:.3f} / "
                f"{item['recall'][3]:.3f} / "
                f"{item['recall'][5]:.3f}"
            )
            print(f"  Reciprocal rank: {item['reciprocal_rank']:.3f}")

    print("\nQueries where the first relevant ticket ranks below 3")
    print("=" * 78)

    for result in results:
        failures = [
            item
            for item in result["query_results"]
            if item["first_relevant_rank"] is None
            or item["first_relevant_rank"] > 3
        ]

        print(f"\n{result['name']}: {len(failures)} queries")

        for item in failures:
            rank = item["first_relevant_rank"]
            rank_text = str(rank) if rank is not None else "not found"

            print(f"  Query: {item['query']}")
            print(f"    First relevant rank: {rank_text}")
            print(f"    Relevant IDs: {item['relevant_ids']}")
            print(f"    Top 5 IDs: {item['ranked_ids'][:5]}")


if __name__ == "__main__":
    main()
