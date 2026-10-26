#!/usr/bin/env bash
# usage: bash scripts/interleave_backends.sh OUTDIR   (alternates pytorch/onnx, 3 rounds, 64 users)
set -u
OUT=${1:?outdir}
cd ~/projects/ticketwise && source .venv/bin/activate
[ -e "$OUT" ] && { echo "$OUT exists; move it first" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then order="pytorch onnx"; else order="onnx pytorch"; fi
  for b in $order; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TICKETWISE_ONNX_THREADS=1 TICKETWISE_MODEL_BACKEND=$b \
    TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 \
      taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers 4 &
    pid=$!
    until curl -sf localhost:8000/health >/dev/null; do
      kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }; sleep 1
    done
    curl -s localhost:8000/health | grep -q "\"model_backend\":\"$b\"" || { echo "WRONG BACKEND" >&2; kill $pid; pkill -f "uvicorn backend"; exit 1; }
    sleep 32
    bash scripts/locust_run.sh "$OUT" ${b}_r$r "/search/semantic /search/semantic/unbatched" "64" 30 >/dev/null
    kill $pid; sleep 5; pkill -f "uvicorn backend" 2>/dev/null; sleep 3
  done
done
OUT="$OUT" python - <<'PY'
import csv, glob, os, re, statistics as st
from collections import defaultdict
g = defaultdict(list)
for f in glob.glob(os.environ["OUT"] + "/*_stats.csv"):
    m = re.match(r"(pytorch|onnx)_r\d+(_search_semantic(?:_unbatched)?)_u64_stats\.csv", os.path.basename(f))
    if not m: continue
    for r in csv.DictReader(open(f)):
        if r["Name"] == "Aggregated":
            g[(m.group(2), m.group(1))].append((float(r["Requests/s"]), float(r["99%"])))
for ep in ("_search_semantic", "_search_semantic_unbatched"):
    p, o = g[(ep, "pytorch")], g[(ep, "onnx")]
    pr, orr = [x[0] for x in p], [x[0] for x in o]
    print(f"{ep:28} pytorch {st.median(pr):6.1f} [{min(pr):.0f}-{max(pr):.0f}] p99 {st.median(x[1] for x in p):4.0f} | "
          f"onnx {st.median(orr):6.1f} [{min(orr):.0f}-{max(orr):.0f}] p99 {st.median(x[1] for x in o):4.0f} | "
          f"onnx/pytorch {st.median(orr)/st.median(pr):.2f}x")
PY
