import os

from sentence_transformers import SentenceTransformer

from backend.app.search import load_tickets


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearch:
    """Search support tickets using configurable inference backends."""

    _triton = None  # set by _init_triton() for the Triton backend

    _triton = None  # set by _init_triton() for the Triton backend

    def __init__(self, backend: str | None = None):
        self.tickets = load_tickets()

        # Keep PyTorch as the default for backward compatibility.
        self.backend = (
            backend or os.getenv("TICKETWISE_MODEL_BACKEND", "pytorch")
        ).lower()

        if self.backend not in {"pytorch", "onnx", "triton"}:
            raise ValueError(
                "Unsupported model backend. Choose 'pytorch', 'onnx' or 'triton'."
            )

        if self.backend == "onnx":
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            threads = int(os.getenv("TICKETWISE_ONNX_THREADS", "0"))
            if threads > 0:
                session_options.intra_op_num_threads = threads
            self.model = SentenceTransformer(
                MODEL_NAME,
                backend="onnx",
                model_kwargs={
                    "provider": "CPUExecutionProvider",
                    "file_name": os.getenv("TICKETWISE_ONNX_FILE", "onnx/model.onnx"),
                    "session_options": session_options,
                },
            )
        else:
            self.model = SentenceTransformer(
                MODEL_NAME,
                device="cpu",
            )

        self._triton = None

        if self.backend == "triton":
            self._init_triton()

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
        query_embeddings = self._encode_queries(queries)

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

    # ---- Triton Inference Server backend ----------------------------------
    # Documents are still embedded in-process at startup (full length). Only
    # query embeddings go to Triton, with queries padded to a fixed length.

    def _init_triton(self) -> None:
        import httpx

        self.triton_model = os.getenv("TICKETWISE_TRITON_MODEL", "minilm_batch")
        self.triton_seq_len = int(os.getenv("TICKETWISE_TRITON_SEQ_LEN", "64"))
        base_url = os.getenv("TICKETWISE_TRITON_URL", "http://127.0.0.1:8100")

        self._triton = httpx.Client(
            base_url=base_url,
            timeout=30.0,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
        )
        self._triton_url = f"/v2/models/{self.triton_model}/infer"

        metadata = self._triton.get(f"/v2/models/{self.triton_model}")
        metadata.raise_for_status()
        self._triton_inputs = [item["name"] for item in metadata.json()["inputs"]]
        # Use the sequence length the Triton model was built for.
        self.triton_seq_len = int(metadata.json()["inputs"][0]["shape"][-1])

    def _encode_queries(self, queries: list[str]):
        """Return normalized query embeddings from the configured backend."""
        if self._triton is not None:
            return self._encode_queries_triton(queries)

        return self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _encode_queries_triton(self, queries: list[str]):
        import numpy as np

        encoded = self.model.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=self.triton_seq_len,
            return_tensors="np",
        )
        shape = list(encoded["input_ids"].shape)
        inputs = []

        for name in self._triton_inputs:
            values = (
                encoded[name] if name in encoded
                else np.zeros_like(encoded["input_ids"])
            )
            inputs.append({
                "name": name,
                "shape": shape,
                "datatype": "INT64",
                "data": np.asarray(values, dtype=np.int64).ravel().tolist(),
            })

        response = self._triton.post(
            self._triton_url,
            json={"inputs": inputs, "outputs": [{"name": "sentence_embedding"}]},
        )
        response.raise_for_status()
        output = response.json()["outputs"][0]

        return np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])
