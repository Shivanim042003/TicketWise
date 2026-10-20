import os

from sentence_transformers import SentenceTransformer

from backend.app.search import load_tickets


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearch:
    """Search support tickets using configurable inference backends."""

    def __init__(self, backend: str | None = None):
        self.tickets = load_tickets()

        # Keep PyTorch as the default for backward compatibility.
        self.backend = (
            backend or os.getenv("TICKETWISE_MODEL_BACKEND", "pytorch")
        ).lower()

        if self.backend not in {"pytorch", "onnx"}:
            raise ValueError(
                "Unsupported model backend. Choose 'pytorch' or 'onnx'."
            )

        if self.backend == "onnx":
            self.model = SentenceTransformer(
                MODEL_NAME,
                backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider"},
            )
        else:
            self.model = SentenceTransformer(
                MODEL_NAME,
                device="cpu",
            )

        self.documents = [
            f"{ticket['title']}. {ticket['description']}"
            for ticket in self.tickets
        ]

        # Precompute ticket embeddings once during initialization.
        self.embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Rank tickets independently for multiple queries in one encode call."""
        if not queries:
            return []

        # Encode all queries together instead of making one model call per query.
        query_embeddings = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Each row corresponds to one query; each column to one ticket.
        # Shape: (number of queries, number of tickets).
        scores_matrix = query_embeddings @ self.embeddings.T

        batch_results = []

        for scores in scores_matrix:
            ranked_indices = scores.argsort()[::-1][:top_k]

            query_results = [
                {
                    **self.tickets[index],
                    "score": round(float(scores[index]), 4),
                }
                for index in ranked_indices
            ]

            batch_results.append(query_results)

        return batch_results

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for one query using the batch-search implementation."""
        return self.search_batch([query], top_k=top_k)[0]
