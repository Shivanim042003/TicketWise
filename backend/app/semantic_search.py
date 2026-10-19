
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

        # Precompute document embeddings once during initialization.
        self.embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Rank tickets by cosine similarity to the query."""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                **self.tickets[index],
                "score": round(float(scores[index]), 4),
            }
            for index in ranked_indices
        ]
