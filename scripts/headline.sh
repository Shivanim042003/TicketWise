#!/usr/bin/env bash
set -u
cd ~/projects/ticketwise && source .venv/bin/activate
[ -e results/headline ] && { echo "results/headline exists; move it first" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 \
  taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers 4 &
pid=$!
until curl -sf localhost:8000/health >/dev/null; do
  kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }; sleep 1
done
sleep 32
for r in 1 2 3; do
  bash scripts/locust_run.sh results/headline h4x1_r$r "/search/semantic /search/semantic/unbatched" "16 24 32 48 64 96 128" 30 >/dev/null
done
kill $pid; sleep 5; pkill -f "uvicorn backend" 2>/dev/null
python - <<'PY'
import csv, glob, statistics as st, re
from collections import defaultdict
g = defaultdict(list)
for f in glob.glob("results/headline/h4x1_r*_stats.csv"):
    m = re.search(r"_r\d+(_search_semantic(?:_unbatched)?)_u(\d+)_stats", f)
    for r in csv.DictReader(open(f)):
        if r["Name"] == "Aggregated":
            g[(m.group(1), int(m.group(2)))].append((float(r["Requests/s"]), float(r["95%"]), float(r["99%"])))
print(f"{'endpoint':32} {'users':>5} {'rps (range)':>16} {'p95':>6} {'p99':>6}")
for (ep, u), v in sorted(g.items()):
    rps = [x[0] for x in v]
    print(f"{ep:32} {u:5d} {st.median(rps):7.1f} [{min(rps):.0f}-{max(rps):.0f}] {st.median(x[1] for x in v):6.0f} {st.median(x[2] for x in v):6.0f}")
PY
