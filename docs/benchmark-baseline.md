# TicketWise Performance Baseline

## Environment and workload

- Search dataset: 5 support tickets
- Load: 100 measured requests per endpoint
- Concurrency: 5
- Device: CPU
- Measurement: end-to-end HTTP latency
- Warm-up: 3 requests per endpoint, excluded from measurements

## Results

| Metric | Keyword | Semantic |
|---|---:|---:|
| p50 latency (ms) | 10.05 | 111.97 |
| p95 latency (ms) | 18.83 | 167.27 |
| p99 latency (ms) | 46.00 | 220.85 |
| Mean latency (ms) | 12.04 | 116.85 |
| Throughput (req/s) | 390.89 | 42.32 |
| Errors | 0 | 0 |
| Error rate (%) | 0.0 | 0.0 |

## Initial observations

Semantic search has higher latency and lower throughput than keyword search
in this workload. Both endpoints completed all 100 measured requests.

This is a preliminary local benchmark on a five-ticket dataset. It does not
establish production capacity or isolate model inference latency.

## Next experiment

Measure query-embedding time separately from similarity calculation and
result sorting. Use the measurements to identify the main optimization
opportunity before trying ONNX Runtime or other serving optimizations.
