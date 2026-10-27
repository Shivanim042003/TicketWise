#!/usr/bin/env bash
# usage: bash scripts/queue_sweep.sh OUTDIR
# For each (mode, queue size, repetition): fresh server, warm-up 48 users 15 s, then
# pre 48 users 20 s -> spike 512 users 30 s -> post 48 users 15 s.
# mode "imm" = clients retry immediately after a 503; "ra" = clients wait 1 s (Retry-After).
# Takes about 30 minutes. Keep the machine awake and close other programs.
set -u
OUT=${1:?outdir}
cd ~/projects/ticketwise && source .venv/bin/activate
[ -e "$OUT" ] && { echo "$OUT exists; move it first" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
mkdir -p "$OUT"
export TARGET=/search/semantic
run() {  # tag users seconds honor
  HONOR_RETRY=$4 taskset -c 4-7 locust -f scripts/locustfile_overload.py --headless -u $2 -r $2 -t ${3}s \
    --processes 3 --reset-stats --host http://127.0.0.1:8000 \
    --csv "$OUT/$1" --only-summary >/dev/null 2>&1
}
one() {  # mode size rep honor
  local tag=$1_q$2_r$3
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TICKETWISE_ONNX_THREADS=1 TICKETWISE_MODEL_BACKEND=onnx \
  TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 TICKETWISE_MAX_QUEUE_SIZE=$2 \
    taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers 4 &
  local pid=$!
  until curl -sf localhost:8000/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }; sleep 1
  done
  run warm 48 15 $4; rm -f "$OUT"/warm_*
  run ${tag}_pre 48 20 $4
  run ${tag}_spike 512 30 $4
  run ${tag}_post 48 15 $4
  kill $pid; sleep 3; pkill -f "uvicorn backend" 2>/dev/null
  while curl -sf localhost:8000/health >/dev/null; do sleep 1; done
  echo "done $tag"
}
sizes=(8 16 32 64)
for rep in 1 2 3; do
  # rotate size order each repetition to limit drift effects
  for k in 0 1 2 3; do one imm ${sizes[$(( (k + rep - 1) % 4 ))]} $rep 0; done
done
for rep in 1 2 3; do
  for s in 16 64; do one ra $s $rep 1; done
done
python - "$OUT" <<'PY'
import csv, glob, re, statistics as st, sys
from collections import defaultdict
out = sys.argv[1]
d = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{out}/*_stats.csv"):
    m = re.search(r"/(imm|ra)_q(\d+)_r(\d+)_(pre|spike|post)_stats\.csv", f)
    if not m: continue
    mode, q, _, ph = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    rows = {r["Name"]: r for r in csv.DictReader(open(f))}
    g = lambda n, k: float(rows[n][k]) if n in rows else 0.0
    key = (mode, q)
    d[key][ph + "_acc"].append(g("accepted", "Requests/s"))
    d[key][ph + "_p99"].append(g("accepted", "99%"))
    d[key][ph + "_p50"].append(g("accepted", "50%"))
    d[key][ph + "_slo"].append(g("within_slo", "Requests/s"))
    d[key][ph + "_rej"].append(g("rejected_503", "Requests/s"))
    d[key][ph + "_err"].append(g("other_error", "Request Count"))
med = lambda k, f: st.median(d[k][f]) if d[k][f] else float("nan")
print("SPIKE phase (512 clients), median of 3 runs")
print(f"{'clients':8} {'queue':>5} {'accepted/s':>10} {'<=200ms/s':>10} {'acc p50':>8} {'acc p99':>8} {'503/s':>8}")
for (mode, q) in sorted(d, key=lambda x: (x[0] != "imm", x[1])):
    k = (mode, q)
    print(f"{'retry-now' if mode=='imm' else 'Retry-After':8} {q:5d} {med(k,'spike_acc'):10.1f} {med(k,'spike_slo'):10.1f} "
          f"{med(k,'spike_p50'):8.0f} {med(k,'spike_p99'):8.0f} {med(k,'spike_rej'):8.1f}")
print("\nNORMAL load (48 clients), median of 3 runs: accepted/s and p99 ms before / after the spike")
print(f"{'clients':8} {'queue':>5} {'pre acc/s':>10} {'pre p99':>8} {'post acc/s':>10} {'post p99':>8} {'errors':>7}")
for (mode, q) in sorted(d, key=lambda x: (x[0] != "imm", x[1])):
    k = (mode, q)
    errs = int(sum(sum(d[k][p + "_err"]) for p in ("pre", "spike", "post")))
    print(f"{'retry-now' if mode=='imm' else 'Retry-After':8} {q:5d} {med(k,'pre_acc'):10.1f} {med(k,'pre_p99'):8.0f} "
          f"{med(k,'post_acc'):10.1f} {med(k,'post_p99'):8.0f} {errs:7d}")
PY
