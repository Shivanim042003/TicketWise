# Cost model: dollars per million searches

Capacity is the highest measured throughput with p99 <= 200 ms (median of repetitions).
Cost per million = (price per hour / 3600) / rps x 1,000,000, divided by a 60% utilization
target in the planning columns. Prices are AWS us-east-1 on-demand: c7i.xlarge $0.1785/h, c7a.xlarge $0.2053/h.

| Setup | Users at SLO | rps at SLO | c7i $/1M (100% util) | c7i $/1M (60% util) | c7a $/1M (60% util) |
|---|---:|---:|---:|---:|---:|
| PyTorch naive (1 process x 4 threads, unbatched) | 8 | 62 | 0.803 | 1.339 | 1.540 |
| PyTorch 4 workers x 1 thread, unbatched | 16 | 240 | 0.207 | 0.344 | 0.396 |
| PyTorch 4 workers x 1 thread, batched | 24 | 406 | 0.122 | 0.204 | 0.234 |
| ONNX 4 workers x 1 thread, unbatched | 48 | 536 | 0.093 | 0.154 | 0.177 |
| ONNX 4 workers x 1 thread, batched | 96 | 852 | 0.058 | 0.097 | 0.112 |

Cheapest: ONNX 4 workers x 1 thread, batched is 13.8x cheaper per search than PyTorch naive (1 process x 4 threads, unbatched).

## Caveats
- Throughput was measured on a laptop (Intel Core Ultra 9 288V, 4 logical CPUs for the server, WSL2),
  not on EC2. A cloud 4 vCPU instance is 2 physical cores with hyperthreads, so absolute rps will differ;
  treat the ratios between setups as more reliable than the dollar values, and re-run the headline
  benchmark on a real instance before quoting costs.
- The PyTorch headline ran in a slower machine state than the ONNX runs (same-session check at 64 users:
  PyTorch batched 606 rps vs 494 in the headline), so PyTorch rows are pessimistic by roughly 20%.
- Excludes load balancer, storage, data transfer, idle capacity and operations cost.
- Prices come from third-party price trackers (checked 2026-10-25); verify against the AWS pricing page.
