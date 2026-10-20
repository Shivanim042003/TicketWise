# TicketWise Retrieval Baseline

## Evaluation dataset
- Synthetic tickets: 25
- Evaluation queries: 25
- Relevance labels: initial synthetic labels; manual review remains necessary.

## Retrieval quality

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Recall@1 | 0.820 | 0.820 |
| Recall@3 | 0.980 | 0.980 |
| Recall@5 | 1.000 | 1.000 |
| MRR | 1.000 | 1.000 |

Full rankings matched for 25 of 25 queries.

## Caveats
- The evaluation set is small and synthetic.
- These results do not establish real-world retrieval quality.
- Dynamic batching must preserve rankings and API behavior.
- Performance comparisons must use the same workload and environment.
