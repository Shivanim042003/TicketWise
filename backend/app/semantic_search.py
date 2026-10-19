
from sentence_transformers import SentenceTransformer

from backend.app.search import load_tickets


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearch:
    """Search support tickets using text embeddings."""

    def __init__(self):
        self.tickets = load_tickets()

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )

        self.documents = [
            f"{ticket['title']}. {ticket['description']}"
            for ticket in self.tickets
        ]

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

        results = []

        for index in ranked_indices:
            results.append({
                **self.tickets[index],
                "score": round(float(scores[index]), 4),
            })

        return results
