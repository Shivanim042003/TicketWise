import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from backend.app.semantic_search import SemanticSearch

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data" / "evaluation_queries_v2.json"
OUTPUT_PATH = ROOT / "data" / "profiling" / "hybrid_retrieval_v3.json"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_features(text):
    words = tokenize(text)
    bigrams = [
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    ]
    return words + bigrams


def lexical_scores(query, documents):
    """TF-IDF-like lexical score using words and adjacent word pairs."""
    tokenized_docs = [lexical_features(doc) for doc in documents]
    query_features = set(lexical_features(query))
    doc_count = len(tokenized_docs)

    document_frequency = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    scores = []
    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        score = 0.0

        for feature in query_features:
            if feature not in frequencies:
                continue

            inverse_frequency = np.log(
                1 + doc_count / (1 + document_frequency[feature])
            )
            score += (
                1 + np.log(frequencies[feature])
            ) * inverse_frequency

        scores.append(score)

    scores = np.asarray(scores, dtype=float)
    maximum = scores.max() if len(scores) else 0.0
    return scores / maximum if maximum > 0 else scores


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant:
            return 1.0 / rank
    return 0.0 if relevant else None


def summarize(rows):
    relevant_rows = [r for r in rows if r["relevant_ticket_ids"]]
    metrics = {"query_count": len(rows)}

    for k in (1, 3, 5):
        values = [
            recall_at_k(r["ranked_ids"], r["relevant_ticket_ids"], k)
            for r in relevant_rows
        ]
        metrics[f"recall_at_{k}"] = float(np.mean(values)) if values else None

    rr = [
        reciprocal_rank(r["ranked_ids"], r["relevant_ticket_ids"])
        for r in relevant_rows
    ]
    metrics["mrr"] = float(np.mean(rr)) if rr else None
    return metrics


def main():
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    engine = SemanticSearch(backend="pytorch")

    documents = [
        f"{ticket['title']}. {ticket['description']}. "
        f"Category: {ticket.get('category', '')}"
        for ticket in engine.tickets
    ]

    ticket_ids = [ticket["id"] for ticket in engine.tickets]
    weights = [1.0, 0.9, 0.8, 0.7, 0.5]
    method_rows = {f"semantic_{w:.1f}": [] for w in weights}

    for item in queries:
        query = item["query"]
        relevant = item["relevant_ticket_ids"]

        semantic_results = engine.search(query, top_k=len(engine.tickets))
        semantic_by_id = {r["id"]: r["score"] for r in semantic_results}
        semantic = np.array([
            semantic_by_id[ticket_id] for ticket_id in ticket_ids
        ])

        lexical = lexical_scores(query, documents)

        for weight in weights:
            combined = weight * semantic + (1.0 - weight) * lexical
            order = np.argsort(combined)[::-1]
            ranked_ids = [ticket_ids[i] for i in order]

            method_rows[f"semantic_{weight:.1f}"].append({
                "query": query,
                "type": item["type"],
                "relevant_ticket_ids": relevant,
                "ranked_ids": ranked_ids,
                "top_score": float(combined[order[0]]),
                "reciprocal_rank": reciprocal_rank(ranked_ids, relevant),
                "recall": {
                    str(k): recall_at_k(ranked_ids, relevant, k)
                    for k in (1, 3, 5)
                },
            })

    output = {
        "query_count": len(queries),
        "lexical_features": "words_and_bigrams",
        "methods": {},
    }

    for method, rows in method_rows.items():
        output["methods"][method] = {
            "overall": summarize(rows),
            "by_type": {
                query_type: summarize([
                    row for row in rows if row["type"] == query_type
                ])
                for query_type in sorted({row["type"] for row in rows})
            },
            "rankings": rows,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    for method, result in output["methods"].items():
        print(f"\n{method.upper()}")
        print(json.dumps(result["overall"], indent=2))
        print("Paraphrases:")
        print(json.dumps(result["by_type"].get("paraphrase"), indent=2))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
