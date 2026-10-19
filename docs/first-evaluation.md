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

## Retrieval evaluation: keyword vs. embeddings

Evaluation set: 5 queries over 5 tickets.
Relevance judgments: manually labeled ticket IDs.
Embedding model: sentence-transformers/all-MiniLM-L6-v2.
Similarity: cosine similarity of normalized embeddings.

| Metric | Keyword search | Embedding search |
|---|---:|---:|
| Mean Recall@1 | 0.800 | 0.800 |
| Mean Recall@3 | 0.900 | 1.000 |
| MRR | 1.000 | 1.000 |

### Interpretation

Embedding search improved Mean Recall@3 by 0.100 on this small evaluation
set. It retrieved both labeled deployment-related tickets in the top three
for the first query, whereas keyword search missed TICK-004.

Mean Recall@1 and MRR were unchanged. Every query's first result was
relevant for both methods, but the evaluation set is too small to establish
general performance superiority.

These are preliminary results on five tickets and five queries.
