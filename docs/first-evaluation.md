# TicketWise — Evaluation Case 1

## Query
API returns 503 after deployment

## Baseline
Method: Exact-token keyword overlap with stop-word filtering.
Stop word: after

## Observed results

| Rank | Ticket ID | Score |
|---|---|---:|
| 1 | TICK-001 | 4 |
| 2 | TICK-003 | 1 |

TICK-004 was not returned because it had no matching query terms.

## Limitation
Keyword overlap misses potentially relevant tickets that describe the same
problem using different words, such as "service unavailable" or "release".

## Next experiment
Compare the same query using cosine similarity between normalized text
embeddings from sentence-transformers/all-MiniLM-L6-v2.

These results are the observed keyword baseline, not embedding results.

## Embedding-search results

Model: sentence-transformers/all-MiniLM-L6-v2
Device: CPU
Embedding dimension: 384
Similarity: cosine similarity using normalized embeddings

| Rank | Ticket ID | Cosine similarity |
|---|---|---:|
| 1 | TICK-001 | 0.850 |
| 2 | TICK-004 | 0.405 |
| 3 | TICK-002 | 0.284 |
| 4 | TICK-003 | 0.168 |
| 5 | TICK-005 | 0.139 |

## Initial observation

Embedding search retrieved TICK-004, "Service unavailable after release",
in second place despite its different wording. The keyword baseline did not
return this ticket.

These scores are similarity values, not probabilities. This is a preliminary
experiment on only five tickets and one query; it does not establish overall
retrieval quality.
