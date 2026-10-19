
import statistics
import time

import numpy as np

from backend.app.semantic_search import SemanticSearch


def measure(function, iterations: int = 100) -> list[float]:
    """Return execution times in milliseconds."""
    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000)

    return timings


def report(name: str, timings: list[float]) -> None:
    print(
        f"{name:30s}"
        f"mean={statistics.mean(timings):8.3f} ms  "
        f"p50={np.percentile(timings, 50):8.3f} ms  "
        f"p95={np.percentile(timings, 95):8.3f} ms"
    )


def main():
    search = SemanticSearch()
    query = "API returns 503 after deployment"

    # Warm up the model before measuring.
    search.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = search.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    report(
        "Query embedding",
        measure(
            lambda: search.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        ),
    )

    report(
        "Similarity calculation",
        measure(lambda: search.embeddings @ query_embedding),
    )

    scores = search.embeddings @ query_embedding

    report(
        "Ranking all tickets",
        measure(lambda: scores.argsort()[::-1]),
    )


if __name__ == "__main__":
    main()
