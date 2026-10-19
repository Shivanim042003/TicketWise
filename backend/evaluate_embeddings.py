
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation_queries.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def main():
    tickets = load_json(TICKETS_PATH)
    queries = load_json(QUERIES_PATH)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    # Encode all ticket documents once.
    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    recall_1_scores = []
    recall_3_scores = []
    reciprocal_rank_scores = []

    print("Embedding Search Evaluation")
    print("=" * 60)

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ticket_ids"])

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # For normalized vectors, dot product equals cosine similarity.
        scores = document_embeddings @ query_embedding

        ranked_indices = scores.argsort()[::-1]
        ranked_tickets = [tickets[i] for i in ranked_indices]
        ranked_ids = [ticket["id"] for ticket in ranked_tickets]

        r1 = recall_at_k(ranked_ids, relevant_ids, 1)
        r3 = recall_at_k(ranked_ids, relevant_ids, 3)
        rr = reciprocal_rank(ranked_ids, relevant_ids)

        recall_1_scores.append(r1)
        recall_3_scores.append(r3)
        reciprocal_rank_scores.append(rr)

        print(f"\nQuery: {query}")
        print(f"Relevant tickets: {sorted(relevant_ids)}")
        print("Ranked results:")

        for rank, index in enumerate(ranked_indices, start=1):
            ticket = tickets[index]
            score = float(scores[index])
            print(
                f"  {rank}. {ticket['id']} | "
                f"{score:.3f} | {ticket['title']}"
            )

        print(f"Recall@1: {r1:.3f}")
        print(f"Recall@3: {r3:.3f}")
        print(f"Reciprocal rank: {rr:.3f}")

    n = len(queries)

    print("\n" + "=" * 60)
    print("Aggregate Metrics")
    print(f"Queries evaluated: {n}")

    if n:
        print(f"Mean Recall@1: {sum(recall_1_scores) / n:.3f}")
        print(f"Mean Recall@3: {sum(recall_3_scores) / n:.3f}")
        print(f"MRR: {sum(reciprocal_rank_scores) / n:.3f}")


if __name__ == "__main__":
    main()
