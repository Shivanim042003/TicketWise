
import json
from pathlib import Path

from backend.app.search import search_tickets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation_queries.json"


def load_evaluation_queries() -> list[dict]:
    """Load queries and their human-labeled relevant ticket IDs."""
    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the fraction of relevant tickets retrieved in top K."""
    if not relevant_ids:
        return 0.0

    retrieved_ids = set(ranked_ids[:k])
    relevant_retrieved = retrieved_ids & relevant_ids

    return len(relevant_retrieved) / len(relevant_ids)


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def main() -> None:
    evaluation_queries = load_evaluation_queries()

    recall_1_scores = []
    recall_3_scores = []
    reciprocal_rank_scores = []

    print("Keyword Search Evaluation")
    print("=" * 60)

    for item in evaluation_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ticket_ids"])

        results = search_tickets(query, top_k=20)
        ranked_ids = [ticket["id"] for ticket in results]

        r1 = recall_at_k(ranked_ids, relevant_ids, k=1)
        r3 = recall_at_k(ranked_ids, relevant_ids, k=3)
        rr = reciprocal_rank(ranked_ids, relevant_ids)

        recall_1_scores.append(r1)
        recall_3_scores.append(r3)
        reciprocal_rank_scores.append(rr)

        print(f"\nQuery: {query}")
        print(f"Relevant tickets: {sorted(relevant_ids)}")
        print(f"Retrieved tickets: {ranked_ids}")
        print(f"Recall@1: {r1:.3f}")
        print(f"Recall@3: {r3:.3f}")
        print(f"Reciprocal rank: {rr:.3f}")

    query_count = len(evaluation_queries)

    print("\n" + "=" * 60)
    print("Aggregate Metrics")
    print(f"Queries evaluated: {query_count}")

    if query_count:
        print(
            f"Mean Recall@1: "
            f"{sum(recall_1_scores) / query_count:.3f}"
        )
        print(
            f"Mean Recall@3: "
            f"{sum(recall_3_scores) / query_count:.3f}"
        )
        print(
            f"MRR: "
            f"{sum(reciprocal_rank_scores) / query_count:.3f}"
        )


if __name__ == "__main__":
    main()
