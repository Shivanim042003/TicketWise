
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation_queries.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_backend(name, model, tickets, queries):
    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    recall_1 = []
    recall_3 = []
    reciprocal_ranks = []
    rankings = {}

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ticket_ids"])

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = document_embeddings @ query_embedding
        indices = scores.argsort()[::-1]
        ranked_ids = [tickets[i]["id"] for i in indices]

        rankings[query] = ranked_ids
        recall_1.append(recall_at_k(ranked_ids, relevant_ids, 1))
        recall_3.append(recall_at_k(ranked_ids, relevant_ids, 3))
        reciprocal_ranks.append(
            reciprocal_rank(ranked_ids, relevant_ids)
        )

    n = len(queries)

    if not n:
        raise ValueError("No evaluation queries found.")

    return {
        "name": name,
        "recall_at_1": sum(recall_1) / n,
        "recall_at_3": sum(recall_3) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "rankings": rankings,
    }


def main():
    tickets = load_json(TICKETS_PATH)
    queries = load_json(QUERIES_PATH)

    print("Loading PyTorch model...")
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
        evaluate_backend(
            "PyTorch",
            pytorch_model,
            tickets,
            queries,
        ),
        evaluate_backend(
            "ONNX",
            onnx_model,
            tickets,
            queries,
        ),
    ]

    print("\nAggregate retrieval comparison")
    print("=" * 70)
    print(f"{'Backend':<12} {'Recall@1':>12} {'Recall@3':>12} {'MRR':>12}")
    print("-" * 70)

    for result in results:
        print(
            f"{result['name']:<12}"
            f"{result['recall_at_1']:>12.3f}"
            f"{result['recall_at_3']:>12.3f}"
            f"{result['mrr']:>12.3f}"
        )

    pytorch_rankings = results[0]["rankings"]
    onnx_rankings = results[1]["rankings"]

    matching_queries = sum(
        pytorch_rankings[query] == onnx_rankings[query]
        for query in pytorch_rankings
    )

    print(
        f"\nIdentical full rankings: "
        f"{matching_queries}/{len(queries)} queries"
    )

    print("\nPer-query ranking comparison")

    for item in queries:
        query = item["query"]
        print(f"\nQuery: {query}")
        print("  PyTorch:", " > ".join(pytorch_rankings[query]))
        print("  ONNX:   ", " > ".join(onnx_rankings[query]))


if __name__ == "__main__":
    main()
