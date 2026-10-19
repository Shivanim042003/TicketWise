
import statistics
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.semantic_search import MODEL_NAME
from backend.app.search import load_tickets


def measure(function, iterations=100):
    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000)

    return timings


def report(name, timings):
    print(
        f"{name:25s}"
        f"mean={statistics.mean(timings):8.3f} ms  "
        f"p50={np.percentile(timings, 50):8.3f} ms  "
        f"p95={np.percentile(timings, 95):8.3f} ms"
    )


def main():
    query = "API returns 503 after deployment"

    tickets = load_tickets()
    documents = [
        f"{ticket['title']}. {ticket['description']}"
        for ticket in tickets
    ]

    print("Loading ONNX model...")
    model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider"},
    )

    # Warm up the model before measuring.
    model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_timings = measure(
        lambda: model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    )

    report("ONNX query embedding", query_timings)

    # Measure ticket embedding generation separately.
    document_timings = measure(
        lambda: model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    )

    report("ONNX 5-ticket encoding", document_timings)

    # Confirm that embeddings have the expected shape.
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Ticket embedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()
