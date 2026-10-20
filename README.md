# TicketWise — Semantic Search and Model Serving Optimization

TicketWise is a semantic-search API for support tickets, focused on efficient model serving, dynamic batching, inference-backend comparison, and retrieval-quality evaluation.

The project explores three main questions:

- How much can dynamic batching improve semantic-search API performance?
- How does ONNX Runtime compare with PyTorch for CPU inference?
- How accurately can semantic search retrieve relevant support tickets, and can hybrid ranking improve results?

All benchmark numbers below are local experimental measurements, not production performance guarantees.

## 1. Architecture

- **API framework:** FastAPI
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Inference backends:** PyTorch CPU and ONNX Runtime CPU
- **Retrieval:** Semantic similarity search over support tickets
- **Batching:** Dynamic batching with a maximum batch size of 8 and a maximum wait time of 5 ms
- **Semantic endpoint:** `/search/semantic`
- **Unbatched comparison:** `/search/semantic/unbatched`
- **Health check:** `/health`
- **Batching metrics:** `/metrics/batching`

The `/health` endpoint reports which model backend the running application loaded.

## 2. Dynamic Batching Benchmark

These results compare `/search/semantic` with `/search/semantic/unbatched`. Each table value is the arithmetic mean of three benchmark runs at the same concurrency level. Each run measured 200 requests per endpoint after three warm-up requests.

| Concurrency | Unbatched throughput | Batched throughput | Throughput speedup | Unbatched p95 | Batched p95 | p95 reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 61.30 req/s | 152.25 req/s | 2.48× | 105.50 ms | 44.05 ms | 58.25% |
| 10 | 46.32 req/s | 205.15 req/s | 4.43× | 280.21 ms | 67.42 ms | 75.94% |

Latency values are arithmetic means of the trial-level p95 percentiles, not percentiles calculated from pooled individual requests.

### Methodology

- 200 measured requests per endpoint per trial.
- Three planned trials at concurrency levels 5 and 10.
- Three warm-up requests per endpoint.
- The batched and unbatched endpoints used the same query and `top_k=3`.
- All reported runs completed with zero request errors.
- An additional concurrency-10 run was excluded from the three-run averages.

### Reproduce the benchmark

Start the API server, activate the virtual environment, and run:

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

Each command benchmarks keyword search, batched semantic search, and unbatched semantic search. Repeat each run to assess variability.

These results describe the tested machine, model, workload, and application configuration. They should not be treated as production capacity guarantees.

## 3. PyTorch vs. ONNX Runtime

TicketWise supports PyTorch CPU and ONNX Runtime CPU inference. The following results summarize three trials per backend at each concurrency level. Each trial measured 100 requests per concurrency level after 10 warm-up requests.

| Concurrency | Backend | Median throughput | Median p50 | Median p95 | Median p99 |
|---:|---|---:|---:|---:|---:|
| 1 | PyTorch | 21.24 req/s | 45.82 ms | 60.44 ms | 67.14 ms |
| 1 | ONNX | 63.67 req/s | 14.44 ms | 25.03 ms | 35.91 ms |
| 5 | PyTorch | 58.51 req/s | 82.44 ms | 119.26 ms | 141.06 ms |
| 5 | ONNX | 123.32 req/s | 36.23 ms | 63.45 ms | 70.98 ms |
| 10 | PyTorch | 111.36 req/s | 88.08 ms | 105.51 ms | 109.32 ms |
| 10 | ONNX | 169.57 req/s | 55.56 ms | 77.04 ms | 79.07 ms |

### Observed ONNX improvement

| Concurrency | Throughput improvement | p95 latency reduction |
|---:|---:|---:|
| 1 | 199.80% | 58.58% |
| 5 | 110.76% | 46.80% |
| 10 | 52.27% | 26.98% |

Throughput improvement is relative to PyTorch median throughput. Positive p95 reduction indicates lower ONNX p95 latency.

All measured requests succeeded in the reported trials. These results are local experiments; repeatability and performance on other workloads or hardware have not been established.

### Configure the inference backend

For PyTorch:

```bash
TICKETWISE_MODEL_BACKEND=pytorch \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

For ONNX Runtime:

```bash
TICKETWISE_MODEL_BACKEND=onnx \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check the loaded backend:

```bash
curl -s http://127.0.0.1:8000/health
```

### Reproduce the API benchmark

Start the server with the desired backend in one terminal. In another terminal, activate the virtual environment and run:

```bash
BENCHMARK_BACKEND=onnx \
  python -m backend.benchmark_api_backends
```

Or:

```bash
BENCHMARK_BACKEND=pytorch \
  python -m backend.benchmark_api_backends
```

The benchmark checks `/health` before sending requests. Results are written to:

- `data/profiling/api_benchmark_onnx.json`
- `data/profiling/api_benchmark_pytorch.json`

Each run overwrites its corresponding result file. Copy successful outputs to separate trial filenames before repeating the benchmark.

After preserving three trials per backend, aggregate them with:

```bash
python -m backend.aggregate_api_benchmarks
```

This generates the aggregate JSON, CSV, and Markdown report under `data/profiling/`.

## 4. Retrieval-Quality Evaluation

TicketWise includes an evaluation script for measuring retrieval quality using labeled queries.

Run:

```bash
python -m backend.evaluate_retrieval
```

The current `data/evaluation_queries_v2.json` evaluation contains 18 queries across paraphrase, hard-negative, and no-match categories.

### Latest semantic-only results

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Recall@1 | 0.361 | 0.361 |
| Recall@3 | 0.806 | 0.806 |
| Recall@5 | 0.861 | 0.861 |
| MRR | 0.792 | 0.792 |
| Identical full rankings | 18/18 | 18/18 |

The evaluation found identical full rankings for all 18 queries across the two backends.

Recall@k measures how much of the labeled relevant-ticket set is retrieved within the top k results, according to the evaluation script. Mean reciprocal rank (MRR) rewards placing a relevant result near the top of the ranking.

No-match queries have no relevant ticket IDs, so recall and MRR are undefined for that category. Rejection behavior should be evaluated separately.

### Important evaluation limitations

- The dataset contains only 18 queries and is too small to establish general production retrieval quality.
- Some relevance labels are provisional and need independent review.
- The reported metrics depend on the chosen labels, ticket corpus, query wording, and evaluation implementation.
- Results should be treated as development measurements, not a guarantee of performance on unseen queries.

The evaluation script writes its report to `data/profiling/retrieval_evaluation_v2.json`.

## 5. Experimental Hybrid Ranking

TicketWise also includes an experimental lexical-semantic ranking comparison:

```bash
python -m backend.evaluate_hybrid_retrieval
```

The experiment combines semantic similarity with lexical word and bigram features using different semantic weights.

| Semantic weight | Recall@1 | Recall@3 | Recall@5 | MRR |
|---:|---:|---:|---:|---:|
| 1.0 — semantic only | 0.361 | 0.806 | 0.861 | 0.804 |
| 0.9 — hybrid | 0.444 | 0.806 | 0.931 | 0.850 |
| 0.8 — hybrid | 0.417 | 0.806 | 0.958 | 0.808 |
| 0.7 — hybrid | 0.333 | 0.806 | 0.917 | 0.764 |
| 0.5 — hybrid | 0.333 | 0.806 | 0.889 | 0.736 |

The 0.9 semantic-weight configuration produced the best Recall@1 and MRR among these tested configurations. The 0.8 configuration produced the highest Recall@5.

These results are exploratory: the evaluation set is small and its labels are not fully independently validated. The hybrid experiment does not establish that hybrid ranking will generalize to new queries. **Semantic-only ranking remains the production behavior; hybrid ranking is experimental.**

The report is written to `data/profiling/hybrid_retrieval_v3.json`.

## 6. Similarity Threshold

Semantic search supports a configurable minimum similarity threshold through the `TICKETWISE_MIN_SIMILARITY` environment variable. The default threshold is `0.4396`.

For example:

```bash
TICKETWISE_MIN_SIMILARITY=0.50 \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The threshold is a development setting calibrated against a small query set. It may reject relevant queries or accept irrelevant ones on other workloads. Evaluate it on a larger, independently labeled dataset before relying on it in production.

## 7. Tests

Run the automated test suite:

```bash
python -m pytest
```

The latest recorded run passed all 26 tests across the API, dynamic batcher, and semantic-search test modules. A Starlette deprecation warning was emitted by the test-client integration; it did not cause test failures.

## 8. Historical Benchmark Summary

An earlier benchmark summary script is also available:

```bash
python -m backend.summarize_dynamic_batching
```

It reads `data/dynamic_batching_trials_v2.jsonl` and writes `data/dynamic_batching_summary.json`. Its results cover concurrency levels 1, 5, 10, and 20.

These historical measurements are separate from the repeated benchmark runs in this README. Do not compare the datasets directly without accounting for their different trial counts and benchmark conditions.

## 9. Limitations and Next Steps

- Benchmark results come from a local development environment.
- Only a small number of trials and query examples have been evaluated.
- API latency includes application and batching overhead; it is not equivalent to model-encoding latency.
- Retrieval labels need independent review and a larger held-out evaluation set.
- Performance depends on hardware, operating system, server load, model, batching configuration, and query workload.
- Production adoption requires additional testing of repeatability, retrieval quality, threshold calibration, and deployment requirements.

ONNX Runtime is the leading candidate based on the current local measurements, while semantic-only retrieval remains the production ranking method pending stronger validation.
