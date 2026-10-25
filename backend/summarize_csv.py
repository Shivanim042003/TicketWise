import csv, statistics, sys
from collections import defaultdict

groups = defaultdict(list)
for path in sys.argv[1:]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups[(int(row["concurrency"]), row["endpoint"])].append(row)

for (c, ep), rows in sorted(groups.items()):
    rps = [float(r["throughput_rps"]) for r in rows]
    p95 = [float(r["p95_ms"]) for r in rows]
    p99 = [float(r["p99_ms"]) for r in rows]
    print(f"c={c:<3} {ep:<28} n={len(rows)}  rps {statistics.median(rps):7.2f} "
          f"[{min(rps):.2f}-{max(rps):.2f}]  p95 {statistics.median(p95):6.1f}  p99 {statistics.median(p99):6.1f} ms")
