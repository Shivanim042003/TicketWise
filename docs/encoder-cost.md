# Quality vs serving cost: MiniLM vs bge-small

Script: `python scripts/bench_encoders.py` (raw output: `results/encoder_cost.json`).

## Setup
- ONNX Runtime CPU, 1 intra-op thread (matches the 4 workers x 1 thread deployment).
- Real queries from `data/evaluation_queries_v4.json`, padded to the longest query in
  each batch. Median of 60 batches per point, 5 rounds, models alternating each round.
- Encode time only (tokenisation excluded from timing); no HTTP, queueing or retrieval.

## Results
| Batch | MiniLM ms/batch | bge ms/batch | MiniLM q/s/core | bge q/s/core | bge cost vs MiniLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.7 | 17.0 | 114 | 59 | 1.95x |
| 8 | 68.5 | 137.7 | 117 | 58 | 2.01x |
| 32 | 290.2 | 592.5 | 110 | 54 | 2.04x |

## Quality-per-cost
- bge-small scores nDCG@10 0.703 vs MiniLM 0.648 on BEIR SciFact
  (`docs/retrieval-public-benchmark.md`): about 8.5% better relative.
- It costs about 2.0x the encode compute (12 layers vs 6).
- On the project's 25-ticket set MiniLM ranked best (`docs/retrieval-evaluation.md`),
  so the in-domain evidence does not favour switching.
- Upper-bound cost estimate: if serving cost scaled fully with encode time, the ONNX
  batched cost model figure (about $0.10 per 1M queries) would rise to at most about
  $0.20. Fixed per-request overhead does not double, so the real increase would be smaller.
  An end-to-end load test with bge-small has not been run.

## Decision
MiniLM stays the default. bge-small is the option to choose when ranking quality on a
large, varied corpus matters more than doubling the encode cost.

## Limitations
- Encode-only timing on one laptop; no end-to-end measurement.
- Quality evidence for bge-small comes from one public dataset (SciFact).
- bge-small needs CLS pooling and a query prefix, which the service does not implement yet.
