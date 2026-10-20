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
