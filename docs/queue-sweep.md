# Queue-size sweep and Retry-After behaviour under a concurrency spike

Script: `bash scripts/queue_sweep.sh results/queue_sweep` (raw CSVs in `results/queue_sweep/`).

## Setup
- ONNX, dynamic batching (max batch 32, max wait 0), 4 workers x 1 thread, server on CPUs 0-3,
  load generator on CPUs 4-7. Each run uses a fresh server: warm-up 48 clients 15 s, then
  48 clients 20 s, a spike to 512 clients 30 s, and 48 clients 15 s.
- Queue limit per worker: 8, 16, 32, 64. Three repetitions each, median reported; the order
  of sizes is rotated per repetition.
- "retry-now" clients send the next request immediately after a 503 (closed loop).
  "Retry-After" clients wait 1 s after a 503, the value the service returns.
  Retry-After runs used queue sizes 16 and 64 only, and ran after all retry-now runs
  (not interleaved), so drift could affect that comparison.
- "Within 200 ms" counts accepted (HTTP 200) requests whose client-observed latency was
  at most 200 ms. The spike is a closed-loop concurrency increase (48 to 512 clients), not a
  controlled arrival-rate increase.

## Spike phase (512 clients)
| Clients | Queue | Accepted req/s | Within 200 ms (req/s) | Accepted p50 (ms) | Accepted p99 (ms) | 503/s |
|---|---:|---:|---:|---:|---:|---:|
| retry-now | 8 | 214.4 | 0.5 | 340 | 590 | 8926 |
| retry-now | 16 | 286.7 | 0.2 | 480 | 760 | 8737 |
| retry-now | 32 | 306.3 | 0.6 | 870 | 1400 | 6698 |
| retry-now | 64 | 326.4 | 0.2 | 1200 | 1800 | 6285 |
| Retry-After | 16 | 655.9 | 412.4 | 190 | 340 | 386 |
| Retry-After | 64 | 644.5 | 0.2 | 580 | 770 | 129 |

## Normal load (48 clients), before and after the spike
| Clients | Queue | Before req/s | Before p99 | After req/s | After p99 | Non-503 errors |
|---|---:|---:|---:|---:|---:|---:|
| retry-now | 8 | 651.3 | 200 | 645.1 | 120 | 0 |
| retry-now | 16 | 764.5 | 100 | 652.9 | 130 | 0 |
| retry-now | 32 | 790.1 | 120 | 786.6 | 130 | 0 |
| retry-now | 64 | 799.2 | 110 | 802.9 | 100 | 0 |
| Retry-After | 16 | 612.3 | 150 | 645.4 | 170 | 0 |
| Retry-After | 64 | 630.7 | 170 | 638.2 | 150 | 0 |

## Findings
- With clients that retry immediately, smaller queues lowered accepted latency
  (p99 1800 ms at queue 64, 590 ms at queue 8) but never reached 200 ms, and accepted
  throughput fell (326 to 214 req/s). Rejection work probably competes for CPU with
  accepted work (not profiled).
- With clients that honour Retry-After, queue 16 kept accepted throughput near normal
  (656 req/s) and delivered 412 req/s within 200 ms; queue 64 delivered about none within
  200 ms. Cooperative clients and a short queue together are what kept requests inside
  the target; neither did alone.
- Latency returned to normal (p99 100-170 ms) in the first 15 s after the spike for
  every setting.

## Limitations
- Closed-loop clients, one laptop, three repetitions per point; normal-load throughput
  differed by about 20% between rows that should be equivalent.
- Retry-After runs were not interleaved with retry-now runs.
- Only queue sizes 8-64 were tested, and only one spike size (512 clients).
- No queue-wait deadlines were tested.
