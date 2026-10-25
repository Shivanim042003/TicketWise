# Serving benchmarks: dynamic batching for semantic search

## Summary
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-d), PyTorch CPU backend, FastAPI + uvicorn.
- With a 4-core server budget, dynamic batching (cap 32, wait 0) gives 1.5-2.5x the throughput of a
  well-configured unbatched baseline (4 workers x 1 thread), and roughly 7-8x a naive single-process
  unbatched server.
- At p99 <= 200 ms: about 400 rps batched (24 users) vs about 240 rps unbatched scale-out (16 users)
  vs about 60 rps for the naive server.

## Setup
- Hardware: Intel Core Ultra 9 288V laptop (8 logical CPUs), WSL2 Ubuntu, plugged in.
- Server pinned to CPUs 0-3; Locust (3 processes) pinned to CPUs 4-7.
- Batcher: one asyncio worker per process, `max_batch_size=32`, `max_wait_ms=0` (takes whatever is
  already queued, never waits).
- Load: Locust closed-loop users, no think time, queries sampled from 25 evaluation queries, `top_k=3`.
  30 s measured after a 15 s unrecorded warm-up. Headline numbers are medians of 3 repetitions.

## Results (4 workers x 1 thread each; batched vs unbatched)

| Users | Batched rps (range) | Batched p99 | Unbatched rps (range) | Unbatched p99 | Ratio |
|---:|---|---:|---|---:|---:|
| 16  | 348 [335-388] | 71 ms  | 240 [157-243] | 160 ms  | 1.45x |
| 24  | 406 [358-426] | 120 ms | 228 [205-302] | 250 ms  | 1.78x |
| 32  | 385 [252-398] | 210 ms | 239 [195-250] | 300 ms  | 1.61x |
| 48  | 435 [432-460] | 240 ms | 264 [239-286] | 390 ms  | 1.65x |
| 64  | 494 [489-562] | 220 ms | 246 [232-268] | 560 ms  | 2.00x |
| 96  | 517 [498-557] | 360 ms | 259 [200-271] | 700 ms  | 2.00x |
| 128 | 585 [535-588] | 420 ms | 233 [223-249] | 1200 ms | 2.51x |

Throughput at 64 users for other process layouts (single 30 s runs):

| Layout | Unbatched | Batched |
|---|---:|---:|
| 1 worker x 4 threads (naive) | 63 | 469 |
| 2 workers x 2 threads | 143 | 578 |
| 4 workers x 1 thread | 297 | 631 |

Single-sample inference does not parallelize across threads: four one-thread workers give about 4.7x the
throughput of one four-thread process before any batching.

## Tuning findings
- `max_wait_ms`: swept 0/3/5/10 ms with interleaved runs. The wait window gave no throughput benefit at
  10+ clients and costs latency at low load (batched vs unbatched at 1 client: 1.22x, 0.92x, 0.80x,
  0.56x for 0, 3, 5, 10 ms). An earlier bug made `max_wait_ms=0` never batch; fixed by draining the
  queue without waiting (regression test added).
- `max_batch_size`: raising the cap from 8 to 16 to 32 raised throughput at high concurrency; cap 8 was
  the limit above about 40 clients.
- Load generator: a single-process `httpx` client saturated at 260-400 rps (the trivial keyword endpoint
  topped out there); Locust reached 1,665 rps on the same endpoint. `httpx`-based numbers above about
  250 rps are lower bounds and are superseded by the Locust results.

## Limitations
- One machine: WSL2 on a hybrid-core laptop CPU. Absolute throughput drifts about 20% between sessions
  (batched 4x1 at 64 users: 631 rps in one run, 494 rps in the headline run). Compare ratios, not rps.
- 30 s windows, 3 repetitions; some ranges overlap (for example batched at 32 users: 252-398 rps).
- Closed-loop load, not a fixed arrival rate; tail latency under open-loop load may differ.
- 25 short evaluation queries; CPU only. Triton is not yet measured.
- No cloud cost model yet.

## Reproduce
```bash
bash scripts/headline.sh                    # 4 workers x 1 thread, 3 repetitions (about 30 min)
bash scripts/baseline_sweep.sh              # worker/thread layouts
bash scripts/locust_run.sh OUTDIR TAG "/search/semantic /search/semantic/unbatched" "16 32 64" 30
```

## ONNX Runtime (C3 unbatched, C4 batched)

Same harness and layout: ONNX Runtime CPU provider, `onnx/model.onnx` (fp32, pinned explicitly), one
intra-op thread per worker, 4 workers, batch cap 32, wait 0. Embeddings match PyTorch (minimum cosine
1.0, maximum absolute difference 1.3e-7 on 6 queries). Medians of 3 repetitions, 30 s each:

| Users | Batched rps (range) | Batched p99 | Unbatched rps (range) | Unbatched p99 | Batching gain |
|---:|---|---:|---|---:|---:|
| 16  | 602 [578-618] | 55 ms  | 519 [517-526] | 66 ms  | 1.16x |
| 24  | 659 [619-675] | 71 ms  | 522 [509-528] | 95 ms  | 1.26x |
| 32  | 746 [661-761] | 69 ms  | 525 [518-526] | 130 ms | 1.42x |
| 48  | 810 [803-814] | 100 ms | 536 [533-553] | 170 ms | 1.51x |
| 64  | 823 [809-830] | 140 ms | 539 [537-556] | 250 ms | 1.53x |
| 96  | 852 [824-861] | 190 ms | 546 [541-557] | 360 ms | 1.56x |
| 128 | 856 [838-866] | 240 ms | 541 [529-552] | 540 ms | 1.58x |

At p99 <= 200 ms, ONNX batched holds up to 96 users (about 850 rps) and ONNX unbatched up to 48 users
(about 535 rps).

### PyTorch vs ONNX, same session (64 users)

Separate headline runs were about an hour apart and the machine drifted (PyTorch batched at 64 users:
494 rps in the PyTorch headline, 606 rps here). The cross-backend comparison therefore comes from an
interleaved run: 3 rounds, backend order alternated each round, server restarted per backend.

| Endpoint | PyTorch rps (range) | ONNX rps (range) | ONNX / PyTorch |
|---|---|---|---:|
| Batched   | 606 [577-630], p99 200 ms | 723 [668-743], p99 200 ms | 1.19x |
| Unbatched | 237 [233-258], p99 620 ms | 520 [484-535], p99 260 ms | 2.19x |

- ONNX unbatched is about 2.2x PyTorch unbatched and reaches about 86% of PyTorch batched.
- Batching gains 2.6x for PyTorch but 1.4x for ONNX at 64 users: ONNX has less per-request overhead to amortize.
- ONNX with batching is about 3x PyTorch scaled-out unbatched (723 vs 237 rps).
- Use ratios, not absolute rps: absolute throughput drifts 10-25% between sessions.
