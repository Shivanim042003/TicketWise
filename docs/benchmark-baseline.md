
# TicketWise Performance Baseline

## 1. Environment and workload

- Search dataset: 5 synthetic support tickets
- Load: 100 measured requests per endpoint
- Concurrency: 5
- Device: CPU
- Measurement: end-to-end HTTP latency
- Warm-up: 3 requests per endpoint, excluded from measurements
- Model: sentence-transformers/all-MiniLM-L6-v2

## 2. Original PyTorch baseline

| Metric | Keyword | Semantic |
|---|---:|---:|
| p50 latency (ms) | 10.05 | 111.97 |
| p95 latency (ms) | 18.83 | 167.27 |
| p99 latency (ms) | 46.00 | 220.85 |
| Mean latency (ms) | 12.04 | 116.85 |
| Throughput (req/s) | 390.89 | 42.32 |
| Errors | 0 | 0 |
| Error rate (%) | 0.0 | 0.0 |

## 3. ONNX Runtime experiment

Backend: ONNX Runtime using CPUExecutionProvider.

| Metric | Keyword | Semantic |
|---|---:|---:|
| p50 latency (ms) | 10.39 | 37.50 |
| p95 latency (ms) | 28.01 | 83.33 |
| p99 latency (ms) | 41.85 | 126.25 |
| Mean latency (ms) | 12.79 | 43.38 |
| Throughput (req/s) | 368.87 | 111.62 |
| Errors | 0 | 0 |
| Error rate (%) | 0.0 | 0.0 |

## 4. Preliminary observations

In the recorded ONNX run, semantic-search p50 latency was 66.5%
lower than in the original PyTorch run. Throughput was approximately
2.64 times higher.

These figures compare separate benchmark runs, not a controlled,
repeated experiment. CPU scheduling, system load, and other runtime
conditions may affect the results. They should be treated as preliminary.

Keyword-search latency also varied between runs, reinforcing the need
for a controlled comparison.

## 5. Retrieval-quality validation

The PyTorch and ONNX implementations produced identical full rankings
for all five evaluation queries.

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Mean Recall@1 | 0.800 | 0.800 |
| Mean Recall@3 | 1.000 | 1.000 |
| MRR | 1.000 | 1.000 |
| Queries with identical rankings | 5/5 | 5/5 |

The evaluation set contains only five synthetic queries and is too small
to establish general retrieval quality.

## 6. Component profiling

The original PyTorch component profile measured:

| Operation | Mean latency (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Query embedding | 11.979 | 8.135 | 14.055 |
| Similarity calculation | 0.001 | 0.001 | 0.001 |
| Ranking all tickets | 0.001 | 0.001 | 0.001 |

A separate ONNX query-embedding profile measured a mean of 9.968 ms,
p50 of 8.079 ms, and p95 of 19.830 ms. A later comparison script
reported different timings, demonstrating why repeated controlled
measurements are necessary.

## 7. Limitations and next steps

- Only five tickets and five evaluation queries are used.
- Results are from a local CPU environment, not production.
- End-to-end HTTP latency includes more than model inference.
- Repeat both backends under the same conditions before claiming a
  stable performance improvement.
- Preserve raw results and record the backend used for every run.
- Expand the dataset and evaluation queries before making broader
  retrieval-quality claims.
