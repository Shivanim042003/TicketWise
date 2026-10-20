
import numpy as np

from backend.app.semantic_search import SemanticSearch


class FakeModel:
    """Small deterministic model for testing without loading a real model."""

    def __init__(self):
        self.calls = []

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        self.calls.append(list(texts))

        vectors = {
            "ticket A": [1.0, 0.0],
            "ticket B": [0.0, 1.0],
            "query A": [1.0, 0.0],
            "query B": [0.0, 1.0],
        }

        return np.array([vectors[text] for text in texts])


def make_search_engine():
    engine = SemanticSearch.__new__(SemanticSearch)
    engine.tickets = [
        {"id": "TICK-A", "title": "Ticket A"},
        {"id": "TICK-B", "title": "Ticket B"},
    ]
    engine.documents = ["ticket A", "ticket B"]
    engine.model = FakeModel()
    engine.embeddings = engine.model.encode(engine.documents)
    engine.model.calls.clear()
    return engine


def test_search_batch_ranks_each_query_independently():
    engine = make_search_engine()

    results = engine.search_batch(["query A", "query B"], top_k=2)

    assert len(results) == 2
    assert [ticket["id"] for ticket in results[0]] == [
        "TICK-A",
        "TICK-B",
    ]
    assert [ticket["id"] for ticket in results[1]] == [
        "TICK-B",
        "TICK-A",
    ]


def test_search_batch_encodes_queries_in_one_model_call():
    engine = make_search_engine()

    engine.search_batch(["query A", "query B"], top_k=1)

    assert engine.model.calls == [["query A", "query B"]]


def test_search_batch_respects_top_k():
    engine = make_search_engine()

    results = engine.search_batch(["query A", "query B"], top_k=1)

    assert all(len(query_results) == 1 for query_results in results)


def test_search_batch_returns_empty_list_for_no_queries():
    engine = make_search_engine()

    assert engine.search_batch([]) == []
    assert engine.model.calls == []


def test_search_preserves_single_query_interface():
    engine = make_search_engine()

    results = engine.search("query A", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "TICK-A"
    assert results[0]["score"] == 1.0
