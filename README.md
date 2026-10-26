# TicketWise — Semantic Search and Model Serving Optimization

[![TicketWise CI](https://github.com/Shivanim042003/TicketWise/actions/workflows/ci.yml/badge.svg)](https://github.com/Shivanim042003/TicketWise/actions/workflows/ci.yml)

TicketWise is a semantic-search API for support tickets, focused on efficient model serving, dynamic batching, inference-backend comparison, and retrieval-quality evaluation.

The project explores three main questions:

- How much can dynamic batching improve semantic-search API performance?
- How does ONNX Runtime compare with PyTorch for CPU inference?
- How accurately can semantic search retrieve relevant support tickets, and can hybrid ranking improve results?

All benchmark numbers below are local experimental measurements, not production performance guarantees.

## Results at a glance

Semantic search with `all-MiniLM-L6-v2` on CPU, server limited to 4 logical CPUs (Intel Core Ultra 9 288V laptop,
WSL2), closed-loop Locust load, p99 latency target of 200 ms. Capacity is the highest measured throughput that
still met the target.

| Setup | Requests/s at p99 <= 200 ms | Cost per 1M searches (c7i.xlarge, 60% utilization) |
|---|---:|---:|
| PyTorch, 1 process x 4 threads, unbatched | 62 | $1.34 |
| PyTorch, 4 workers x 1 thread, unbatched | 240 | $0.34 |
| PyTorch, 4 workers x 1 thread, dynamic batching | 406 | $0.20 |
| ONNX Runtime, 4 workers x 1 thread, unbatched | 536 | $0.15 |
| **ONNX Runtime, 4 workers x 1 thread, dynamic batching** | **852** | **$0.10** |

- **Scale out before batching.** Four one-thread workers give about 4x the throughput of one four-thread process.
- **Dynamic batching adds about 1.6-1.7x capacity** at the latency target on top of a well-configured baseline.
- **ONNX Runtime is about 2.2x faster than PyTorch without batching and about 1.2x with batching**
  (same-session test, 64 users).
- **Triton Inference Server (CPU, separate container) reached about 70% of in-process ONNX serving** on the same 4 CPUs
  (453 vs 627 req/s with batching at 64 users): the extra network hop costs more than Triton's batching saves here.
- Caveats: one machine, absolute throughput drifts 10-25% between sessions (the PyTorch rows ran in a slower
  state, so they are roughly 20% pessimistic), and the costs are laptop-derived estimates, not EC2 measurements.
  Ratios are more reliable than absolute values.

Details: [docs/serving-benchmarks.md](docs/serving-benchmarks.md) (method, tables, limitations) and
[docs/cost-model.md](docs/cost-model.md) (cost model).

## Quick start

### Local setup

Use Python 3.12. From the repository root, create and activate a virtual environment, install dependencies, and run tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest
```

If your existing virtual environment has no pip, repair or recreate it before installing dependencies.

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

> **Superseded.** This preliminary table used a single-process `httpx` client and compared against a
> naive single-process unbatched server (1 worker x 4 torch threads). The client capped throughput, and the
> baseline was not scaled out. Current results use a Locust load generator and a scaled-out unbatched
> baseline (4 workers x 1 thread): batching gives about 1.7x capacity at p99 <= 200 ms and 2-2.5x at
> saturation. See [docs/serving-benchmarks.md](docs/serving-benchmarks.md).

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

> **Superseded.** These measurements used a single-process `httpx` load generator and a naive single-process
> server. For the current comparison (Locust, scaled-out workers, dynamic batching, same-session test) see the
> ONNX Runtime section of [docs/serving-benchmarks.md](docs/serving-benchmarks.md): ONNX is about 2.2x faster
> than PyTorch unbatched and about 1.2x with dynamic batching.

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

TicketWise evaluates semantic retrieval using labeled support-ticket queries. The evaluator supports selecting a dataset with the `TICKETWISE_EVAL_DATASET` environment variable.

### Run the evaluations

Run the default 18-query v2 evaluation:

    python -m backend.evaluate_retrieval

Run the 60-query v4 evaluation:

    TICKETWISE_EVAL_DATASET=data/evaluation_queries_v4.json python -m backend.evaluate_retrieval

Reports are saved under `data/profiling/`:

- v2: `retrieval_evaluation_evaluation_queries_v2.json`
- v4: `retrieval_evaluation_evaluation_queries_v4.json`

### v2 results: 18 queries

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Recall@1 | 0.361 | 0.361 |
| Recall@3 | 0.806 | 0.806 |
| Recall@5 | 0.861 | 0.861 |
| MRR | 0.792 | 0.792 |
| Identical full rankings | 18/18 | 18/18 |

### v4 results: 60 queries

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Recall@1 | 0.567 | 0.567 |
| Recall@3 | 0.946 | 0.946 |
| Recall@5 | 0.983 | 0.983 |
| MRR | 1.000 | 1.000 |
| Identical full rankings | 60/60 | 60/60 |

The datasets differ in size and composition, so these results should not be interpreted as a direct improvement from v2 to v4. The v4 dataset contains 25 paraphrase, 15 hard-negative, and 20 no-match queries.

Recall@k measures the fraction of labeled relevant ticket IDs retrieved in the top k results. Mean reciprocal rank (MRR) measures the rank of the first relevant result. With multiple relevant ticket IDs per query, MRR can be 1.0 even when Recall@1 is below 1.0.

### Similarity threshold experiment

The semantic endpoints use a configurable minimum similarity threshold controlled by `TICKETWISE_MIN_SIMILARITY` (default: `0.4396`). On the current v4 dataset, this threshold accepted all 40 queries labeled as relevant and rejected all 20 labeled no-match queries.

This is a development-set result, not an estimate of production accuracy. The threshold was selected using earlier evaluation data, and the query labels are provisional.

### Limitations

- Both datasets are small, and some relevance labels require independent review.
- Results depend on query wording, relevance labels, the ticket corpus, and the evaluation implementation.
- Identical PyTorch and ONNX rankings demonstrate consistency on these test queries, not equivalent performance on every possible workload.
- No-match queries have no relevant ticket IDs, so Recall@k and MRR are undefined for that category; rejection behavior is measured separately.
- Validate retrieval quality and threshold behavior on a larger, independently labeled held-out dataset before production use.

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

The latest recorded local run passed all 34 tests across the API, benchmark response-validation, dynamic batcher, and semantic-search modules. GitHub Actions checks syntax and runs the test suite on pushes to main and pull requests.

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

- Serving results come from one laptop (WSL2, hybrid-core CPU). Absolute throughput drifts 10-25% between sessions,
  so ratios are more reliable than absolute numbers, and the cloud costs are laptop-derived estimates.
- Load tests are closed-loop, CPU only, with 25 short queries. Open-loop arrival rates, longer inputs and GPUs
  were not tested. Triton was measured at a single load level (64 users) and not tuned.
- The Docker image was smoke-tested but never load-tested; all benchmarks ran outside a container (see Docker below).

Semantic-only retrieval remains the production ranking method pending stronger validation.

## Serving performance
Dynamic batching on a 4-core CPU budget gives about 1.5-2.5x the throughput of a well-configured
unbatched scale-out baseline and about 7-8x a naive single-process server, at p99 <= 200 ms
(about 400 vs 240 vs 60 req/s). Full method, tables and limitations: [docs/serving-benchmarks.md](docs/serving-benchmarks.md).

ONNX Runtime (CPU, fp32) is about 2.2x faster than PyTorch without batching and about 1.2x faster with
batching (same-session, 64 users). ONNX with dynamic batching is the best configuration measured, at
roughly 3x a scaled-out unbatched PyTorch server. Details: [docs/serving-benchmarks.md](docs/serving-benchmarks.md).

## Docker
`Dockerfile`, `compose.yaml` and `.dockerignore` describe a CPU-only container (4 uvicorn workers x 1 thread,
batch cap 32, wait 0, PyTorch by default; set `TICKETWISE_MODEL_BACKEND=onnx` for ONNX Runtime). Run it with `docker compose up -d`
(the first start downloads the model into a Docker volume; add `TICKETWISE_MODEL_BACKEND=onnx` for ONNX Runtime).
The image builds and starts, and `/health` and `/search/semantic` were smoke-tested with both backends. It has not
been load-tested: all benchmark results in this repository were measured with uvicorn directly under WSL2, not in
a container.

## Retrieval evaluation summary
- Held-out split on the project's own tickets (BM25 vs MiniLM vs bge-small): `docs/retrieval-evaluation.md`.
- Public benchmark (BEIR SciFact, 5,183 docs, 300 queries): `docs/retrieval-public-benchmark.md`.
  bge-small scored nDCG@10 0.703 vs MiniLM 0.648 and BM25 0.664. The serving cost of
  switching models has not been measured.
