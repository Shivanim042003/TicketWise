
import json
import re
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "tickets.json"

# Common words that contribute little to keyword relevance.
STOP_WORDS = {"after"}


def load_tickets() -> list[dict]:
    """Load support tickets from the JSON dataset."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> set[str]:
    """Convert text into lowercase, whole-word tokens."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return tokens - STOP_WORDS


def search_tickets(query: str, top_k: int = 5) -> list[dict]:
    """
    Rank tickets by the number of query terms found in their text.

    This is a simple keyword-search baseline, not semantic search.
    """
    tickets = load_tickets()
    query_terms = tokenize(query)

    results = []

    for ticket in tickets:
        searchable_text = (
            ticket["title"] + " " + ticket["description"]
        )

        ticket_terms = tokenize(searchable_text)
        score = len(query_terms & ticket_terms)

        if score > 0:
            results.append({
                **ticket,
                "score": score,
            })

    # Sort by score from highest to lowest.
    results.sort(key=lambda ticket: ticket["score"], reverse=True)

    return results[:top_k]

