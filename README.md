
## Dynamic Batching Performance

TicketWise supports dynamic batching for semantic search. Concurrent
queries are collected into bounded batches and encoded together using
Sentence Transformers.

### Benchmark Results

The following results compare batched and unbatched semantic search
using 200 measured requests per endpoint per trial, with two trials
at each concurrency level.

| Concurrency | Unbatched Throughput | Batched Throughput | Speedup | p95 Latency Reduction |
|---:|---:|---:|---:|---:|
| 1 | 70.75 req/s | 47.08 req/s | 0.67x | -40.81% |
| 5 | 77.54 req/s | 174.93 req/s | 2.26x | 52.18% |
| 10 | 66.84 req/s | 266.00 req/s | 3.98x | 79.53% |
| 20 | 55.98 req/s | 339.21 req/s | 6.06x | 86.71% |

### Observed Batch Sizes

| Concurrency | Average Batch Size |
|---:|---:|
| 1 | 1.00 |
| 5 | 5.00 |
| 10 | 5.06 |
| 20 | 7.69 |

### Methodology and Limitations

- Each trial measured 200 requests per endpoint after five warm-up requests.
- Two trials were performed at each concurrency level, alternating
  which endpoint ran first.
- The batch size was capped at eight queries.
- All reported trials completed with zero request errors.
- Results were collected locally and should not be treated as
  production performance guarantees.
- These measurements describe the current workload and environment;
  additional trials and larger, representative datasets are needed
  to establish general performance characteristics.

To reproduce the benchmark summary:

```bash
python -m backend.summarize_dynamic_batching
```
