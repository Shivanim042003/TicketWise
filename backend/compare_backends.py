
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.semantic_search import MODEL_NAME
from backend.app.search import load_tickets


QUERY = "API returns 503 after deployment"
ITERATIONS = 100


def benchmark(model, query):
    # Warm up before collecting measurements.
    for _ in range(5):
        model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    timings = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        timings.append((time.perf_counter() - start) * 1000)

    return {
        "mean": float(np.mean(timings)),
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
    }


def get_ranking(model, tickets, query):
    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scores = document_embeddings @ query_embedding
    indices = scores.argsort()[::-1]

    return [
        (tickets[i]["id"], round(float(scores[i]), 4))
        for i in indices
    ]


def main():
    tickets = load_tickets()

    print("Loading PyTorch model...")
    pytorch = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Loading ONNX model...")
    onnx = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider"},
    )

    for name, model in [("PyTorch", pytorch), ("ONNX", onnx)]:
        print(f"\n{name} latency:")
        for metric, value in benchmark(model, QUERY).items():
            print(f"  {metric}: {value:.3f} ms")

    pytorch_ranking = get_ranking(pytorch, tickets, QUERY)
    onnx_ranking = get_ranking(onnx, tickets, QUERY)

    print("\nPyTorch ranking:")
    for ticket_id, score in pytorch_ranking:
        print(f"  {ticket_id}: {score}")

    print("\nONNX ranking:")
    for ticket_id, score in onnx_ranking:
        print(f"  {ticket_id}: {score}")

    same_order = (
        [ticket_id for ticket_id, _ in pytorch_ranking]
        == [ticket_id for ticket_id, _ in onnx_ranking]
    )

    print(f"\nRanking order identical: {same_order}")


if __name__ == "__main__":
    main()
