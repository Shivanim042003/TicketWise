# Public benchmark: BM25 vs MiniLM vs bge-small on BEIR SciFact

Script: `python scripts/eval_public.py` (raw output: `results/retrieval_public_scifact.json`).

## Setup
- Dataset: BEIR SciFact, 5,183 documents, 300 test queries; relevance labels written
  by the dataset authors, not by this project.
- Models: BM25 (own implementation), all-MiniLM-L6-v2 (current model),
  BAAI/bge-small-en-v1.5 (with its query prefix). Document length capped at 256 tokens.
- Sanity check: BM25 (0.664) and MiniLM (0.648) nDCG@10 are within about 1 point of
  commonly published SciFact figures, so the evaluation code is behaving.

## Results (300 test queries)
| Model | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.719 | 0.782 | 0.634 | 0.664 |
| MiniLM (current) | 0.741 | 0.788 | 0.607 | 0.648 |
| bge-small | 0.756 | 0.816 | 0.674 | 0.703 |

## Findings
- On this larger, independently labelled corpus bge-small ranks best, about 4 nDCG@10
  points above MiniLM. On the 25-ticket set (`docs/retrieval-evaluation.md`) MiniLM
  ranked best. The small synthetic set alone would have given a misleading answer.
- MiniLM is only marginally ahead of BM25 on Recall@5 and behind it on MRR and nDCG here.
- bge-small has 12 transformer layers against MiniLM's 6, so it likely costs more to
  serve. The serving cost has not been measured yet, so no quality-per-cost
  recommendation is made.

## Limitations
- SciFact is scientific claim verification, not support tickets. It tests general
  retrieval quality on a real corpus, not performance on support text.
- A single dataset, a single run, no confidence intervals.
