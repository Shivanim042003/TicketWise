# Held-out retrieval evaluation

Script: `python scripts/eval_extended.py` (raw output: `results/retrieval_extended.json`).

## Setup
- Corpus: 25 synthetic tickets (`data/tickets.json`).
- Queries: 60 from `evaluation_queries_v4.json` (paraphrase, hard-negative, no-match).
- Split: 50/50 dev/test, stratified by query type, fixed seed 13. The no-match
  threshold is tuned on dev only and applied to test.
- Models: BM25 (own implementation, k1=1.5, b=0.75), all-MiniLM-L6-v2 (current),
  BAAI/bge-small-en-v1.5 (with its query prefix).
- Test set: 30 queries, 20 with relevant tickets and 10 no-match.

## Test results
| Model | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@10 | No-match acc |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.575 | 0.858 | 0.925 | 1.000 | 0.937 | 0.967 |
| MiniLM (current) | 0.575 | 0.950 | 0.967 | 1.000 | 0.985 | 1.000 |
| bge-small | 0.575 | 0.833 | 0.925 | 1.000 | 0.958 | 1.000 |

## Reading the results
- All three models put a relevant ticket first for every test query (MRR@10 = 1.0).
  Recall@1 is capped because many queries have two relevant tickets.
- MiniLM ranks best on Recall@3, Recall@5 and nDCG@10, which supports keeping it.
- With 20 scored test queries, one query changes a metric by about 5 points, so the
  differences between models are suggestive, not statistically established.

## Limitations
- Only 25 synthetic tickets: ranking a corpus this small is easy, and queries share
  vocabulary with tickets, which flatters BM25.
- Labels were written by the project author, not independently reviewed.
- Next step: evaluate on a larger public corpus with labels written by others.
