# TicketWise — Dynamic Batching Performance

TicketWise implements dynamic batching for semantic search. Concurrent queries are collected into bounded batches and encoded together using Sentence Transformers.

## Benchmark Results

The following results compare `/search/semantic` with `/search/semantic/unbatched`. Each result is the arithmetic mean of three benchmark runs at the same concurrency level. Each run measured 200 requests per endpoint after three warm-up requests.

| Concurrency | Unbatched Throughput | Batched Throughput | Throughput Speedup | Unbatched p95 | Batched p95 | p95 Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 61.30 req/s | 152.25 req/s | 2.48× | 105.50 ms | 44.05 ms | 58.25% |
| 10 | 46.32 req/s | 205.15 req/s | 4.43× | 280.21 ms | 67.42 ms | 75.94% |

Latency figures are arithmetic means of the trial-level percentiles, not percentiles calculated from pooled individual requests.

## Methodology

- 200 measured requests per endpoint per trial.
- Three planned trials at each concurrency level, with three warm-up requests per endpoint.
- Concurrency levels of 5 and 10 were tested.
- Batched and unbatched semantic-search endpoints used the same query and `top_k=3`.
- All reported runs completed with zero request errors.
- An additional concurrency-10 run was performed but excluded from the three-run averages above.
- Results were collected locally and should not be treated as production performance guarantees.

## Interpretation and Limitations

Dynamic batching improved semantic-search throughput in these experiments, with a larger observed advantage at concurrency 10. It also reduced mean p95 latency at both tested concurrency levels.

These measurements describe the current machine, model, workload, and application configuration. Results may vary with hardware, model choice, dataset size, query distribution, batch settings, and server load. Further testing with representative workloads is needed before making production capacity claims.

## Reproducing the Benchmark

Start the TicketWise API server, activate the project virtual environment, and run:

```bash
python -m backend.benchmark \
  --requests 200 \
  --concurrency 5 \
  --output data/profiling/reproduction_c5.csv

python -m backend.benchmark \
  --requests 200 \
  --concurrency 10 \
  --output data/profiling/reproduction_c10.csv
```

Each command benchmarks keyword search, batched semantic search, and unbatched semantic search. Run each command multiple times to assess variability.
## Historical Benchmark Summary

The repository also includes a summary script for an earlier benchmark dataset:

```bash
python -m backend.summarize_dynamic_batching
```

This script reads `data/dynamic_batching_trials_v2.jsonl` and writes `data/dynamic_batching_summary.json`. Its results cover concurrency levels 1, 5, 10, and 20.

These historical results are separate from the repeated benchmark runs documented in the main results table above. Do not compare the two sets directly without accounting for differences in trials and benchmark conditions.

---

## Model Serving Optimization: PyTorch vs. ONNX Runtime

TicketWise supports two CPU inference backends for semantic search:
PyTorch and ONNX Runtime. This experiment compares their API-level
performance while keeping the model, evaluation queries, and application
configuration consistent.

### Architecture

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Inference backends:** PyTorch CPU and ONNX Runtime CPU
- **API framework:** FastAPI
- **Batching:** Dynamic batching with a maximum batch size of 8 and a
  maximum wait time of 5 ms
- **Semantic endpoint:** `/search/semantic`
- **Unbatched comparison endpoint:** `/search/semantic/unbatched`
- **Backend verification:** `/health` reports the backend loaded by the
  running application.
- **Batching metrics:** `/metrics/batching`

Select a backend when starting the server:

```bash
TICKETWISE_MODEL_BACKEND=pytorch \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

```bash
TICKETWISE_MODEL_BACKEND=onnx \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check the loaded backend before benchmarking:

```bash
curl -s http://127.0.0.1:8000/health
```

### API Benchmark Results

The following results summarize three trials per backend at each
concurrency level. Each trial contains 100 measured requests per
concurrency level after 10 warm-up requests.

| Concurrency | Backend | Median throughput | Median p50 | Median p95 | Median p99 |
|---:|---|---:|---:|---:|---:|
| 1 | PyTorch | 21.24 req/s | 45.82 ms | 60.44 ms | 67.14 ms |
| 1 | ONNX | 63.67 req/s | 14.44 ms | 25.03 ms | 35.91 ms |
| 5 | PyTorch | 58.51 req/s | 82.44 ms | 119.26 ms | 141.06 ms |
| 5 | ONNX | 123.32 req/s | 36.23 ms | 63.45 ms | 70.98 ms |
| 10 | PyTorch | 111.36 req/s | 88.08 ms | 105.51 ms | 109.32 ms |
| 10 | ONNX | 169.57 req/s | 55.56 ms | 77.04 ms | 79.07 ms |

### Observed ONNX Improvement

| Concurrency | Throughput improvement | p95 latency reduction |
|---:|---:|---:|
| 1 | 199.80% | 58.58% |
| 5 | 110.76% | 46.80% |
| 10 | 52.27% | 26.98% |

Throughput improvement is calculated relative to PyTorch median throughput.
Positive p95 latency reduction indicates lower ONNX p95 latency.

All measured API requests succeeded in the three trials for both backends.
These results are local experimental measurements, not production
capacity guarantees.

### Retrieval Quality

On the current evaluation set of 25 labeled queries, both backends
produced identical rankings for all 25 queries.

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Recall@1 | 0.820 | 0.820 |
| Recall@3 | 0.980 | 0.980 |
| Recall@5 | 1.000 | 1.000 |
| Identical full rankings | 25/25 | 25/25 |

This indicates that the tested backends preserved retrieval results on
this evaluation set. The small dataset does not establish equivalent
quality for every possible query or production workload.

### Reproducing the API Benchmark

Start the server in one terminal using the desired backend. In a second
terminal, activate the virtual environment and run the benchmark.

For ONNX:

```bash
BENCHMARK_BACKEND=onnx \
  python -m backend.benchmark_api_backends
```

For PyTorch:

```bash
BENCHMARK_BACKEND=pytorch \
  python -m backend.benchmark_api_backends
```

The benchmark verifies the loaded backend through `/health` before
sending measured requests. Results are written to:

- `data/profiling/api_benchmark_onnx.json`
- `data/profiling/api_benchmark_pytorch.json`

The benchmark overwrites the corresponding backend result file on each
run. Copy the JSON file to a separate trial filename after each
successful run to preserve individual measurements.

After preserving three trials per backend, aggregate them with:

```bash
python -m backend.aggregate_api_benchmarks
```

This generates:

- `data/profiling/api_benchmark_aggregate.json`
- `data/profiling/api_benchmark_aggregate.csv`
- `data/profiling/api_benchmark_report.md`

### Limitations and Next Steps

- Only three trials were conducted per backend and concurrency level.
- Trial-to-trial variation was observed, especially for PyTorch at
  concurrency 5.
- Results depend on the hardware, operating system, server load, model,
  dataset, batching configuration, and query workload.
- The evaluation set contains only 25 labeled queries.
- API latency includes application and batching overhead; it is not
  equivalent to model encoding latency.
- Further controlled testing and representative workloads are needed
  before making production performance claims.

ONNX is the leading candidate based on these initial measurements.
Production adoption should follow additional validation of repeatability,
retrieval quality, and deployment requirements.
