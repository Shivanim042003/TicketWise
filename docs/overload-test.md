# Overload and recovery test

Script: `bash scripts/overload_run.sh onnx results/overload` (raw CSVs in `results/overload/`).

## Setup
- ONNX backend, dynamic batching (max batch 32, max wait 0), 4 workers x 1 thread,
  server pinned to CPUs 0-3, load generator on CPUs 4-7.
- Queue limit set to 64 per worker for this test (service default is 256), because with
  512 users each worker holds about 128 requests and a 256 queue would never fill.
- Closed-loop Locust users with no backoff. Phases run back to back against one server:
  48 users 30 s, then 512 users 30 s, then 48 users for 3 x 15 s.
- Accepted requests (HTTP 200) and rejections (HTTP 503) are recorded separately.

## Results
| Phase | Users | Accepted req/s | Accepted p50 (ms) | Accepted p99 (ms) | 503/s |
|---|---:|---:|---:|---:|---:|
| Before | 48 | 485 | 110 | 160 | 0 |
| Spike | 512 | 358 | 1100 | 1500 | 7277 |
| After 1 (15 s) | 48 | 684 | 63 | 160 | 0 |
| After 2 (15 s) | 48 | 712 | 64 | 130 | 0 |
| After 3 (15 s) | 48 | 784 | 54 | 120 | 0 |

Server memory (all processes): 2510 MB at start, 2530 MB peak, 2530 MB at end.
No responses other than 200 and 503.

## Findings
- Under a concurrency spike from 48 to 512 closed-loop clients (about 10x) the service stayed up, kept memory flat, and returned explicit 503s.
  Accepted-request latency was bounded (p99 about 1.5 s) rather than growing without limit.
- Latency was back to normal in the first 15 s window after the spike.
- Rejecting is not free: with about 7,300 rejections per second, accepted throughput
  fell to about half of normal capacity. These clients retry immediately; clients that
  honour `Retry-After` would produce far fewer rejections.
- A 64-entry queue allows about 0.7 s of queueing per worker at the overloaded service
  rate, far above a 200 ms target. The queue limit should be derived from the latency
  budget (a smaller value); this was not tested here.

## Limitations
- One run, one laptop, one queue size. The "before" phase ran slower than the "after"
  phases, probably a cold-start effect, so recovery is judged on latency returning to
  normal, not on matching the first phase.
- Closed-loop clients without backoff, which is harsher than typical clients.
- The spike raises client concurrency in a closed loop; it is not a controlled arrival-rate increase, so achieved request rates differ between phases. Accepted-request p99 (about 1.5 s) stayed far above a 200 ms target: the protection preserves availability, not the latency target.
