#!/usr/bin/env python3
"""Cost per million searches at the p99 SLO, computed from the raw Locust CSVs."""
import csv
import os
import re
import statistics as st
from collections import defaultdict

SLO_P99_MS = 200.0
UTILIZATION = 0.6
PRICES = {"c7i.xlarge": 0.1785, "c7a.xlarge": 0.2053}  # USD/hour, us-east-1 on-demand

SETUPS = [
    ("PyTorch naive (1 process x 4 threads, unbatched)", "results/baseline",
     r"w1t4_search_semantic_unbatched_u(\d+)_stats\.csv"),
    ("PyTorch 4 workers x 1 thread, unbatched", "results/headline",
     r"h4x1_r\d+_search_semantic_unbatched_u(\d+)_stats\.csv"),
    ("PyTorch 4 workers x 1 thread, batched", "results/headline",
     r"h4x1_r\d+_search_semantic_u(\d+)_stats\.csv"),
    ("ONNX 4 workers x 1 thread, unbatched", "results/onnx_headline",
     r"h4x1_r\d+_search_semantic_unbatched_u(\d+)_stats\.csv"),
    ("ONNX 4 workers x 1 thread, batched", "results/onnx_headline",
     r"h4x1_r\d+_search_semantic_u(\d+)_stats\.csv"),
]


def load(directory, pattern):
    by_users = defaultdict(list)
    if not os.path.isdir(directory):
        return by_users
    for name in sorted(os.listdir(directory)):
        m = re.fullmatch(pattern, name)
        if not m:
            continue
        with open(os.path.join(directory, name), newline="") as f:
            for row in csv.DictReader(f):
                if row["Name"] == "Aggregated":
                    by_users[int(m.group(1))].append(
                        (float(row["Requests/s"]), float(row["99%"]))
                    )
    return by_users


def capacity(by_users):
    """Highest median rps among user levels whose median p99 meets the SLO."""
    best = None
    for users, vals in sorted(by_users.items()):
        rps = st.median(v[0] for v in vals)
        p99 = st.median(v[1] for v in vals)
        if p99 <= SLO_P99_MS and (best is None or rps > best[1]):
            best = (users, rps, p99)
    return best


def cost_per_million(price_per_hour, rps):
    return price_per_hour / 3600 / rps * 1e6


lines = [
    "# Cost model: dollars per million searches",
    "",
    f"Capacity is the highest measured throughput with p99 <= {SLO_P99_MS:.0f} ms (median of repetitions).",
    f"Cost per million = (price per hour / 3600) / rps x 1,000,000, divided by a {UTILIZATION:.0%} utilization",
    "target in the planning columns. Prices are AWS us-east-1 on-demand: "
    + ", ".join(f"{k} ${v}/h" for k, v in PRICES.items()) + ".",
    "",
    "| Setup | Users at SLO | rps at SLO | c7i $/1M (100% util) | c7i $/1M (60% util) | c7a $/1M (60% util) |",
    "|---|---:|---:|---:|---:|---:|",
]
costs = {}
for name, directory, pattern in SETUPS:
    cap = capacity(load(directory, pattern))
    if cap is None:
        lines.append(f"| {name} | none | none | - | - | - |")
        continue
    users, rps, _ = cap
    c100 = cost_per_million(PRICES["c7i.xlarge"], rps)
    c60 = c100 / UTILIZATION
    a60 = cost_per_million(PRICES["c7a.xlarge"], rps) / UTILIZATION
    costs[name] = c60
    lines.append(f"| {name} | {users} | {rps:.0f} | {c100:.3f} | {c60:.3f} | {a60:.3f} |")

if len(costs) > 1:
    worst, best = max(costs, key=costs.get), min(costs, key=costs.get)
    lines += ["", f"Cheapest: {best} is {costs[worst] / costs[best]:.1f}x cheaper per search than {worst}."]

lines += [
    "",
    "## Caveats",
    "- Throughput was measured on a laptop (Intel Core Ultra 9 288V, 4 logical CPUs for the server, WSL2),",
    "  not on EC2. A cloud 4 vCPU instance is 2 physical cores with hyperthreads, so absolute rps will differ;",
    "  treat the ratios between setups as more reliable than the dollar values, and re-run the headline",
    "  benchmark on a real instance before quoting costs.",
    "- The PyTorch headline ran in a slower machine state than the ONNX runs (same-session check at 64 users:",
    "  PyTorch batched 606 rps vs 494 in the headline), so PyTorch rows are pessimistic by roughly 20%.",
    "- Excludes load balancer, storage, data transfer, idle capacity and operations cost.",
    "- Prices come from third-party price trackers (checked 2026-10-25); verify against the AWS pricing page.",
]
text = "\n".join(lines) + "\n"
print(text)
with open("docs/cost-model.md", "w", encoding="utf-8") as f:
    f.write(text)
