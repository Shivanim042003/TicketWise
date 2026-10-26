#!/usr/bin/env bash
# usage: bash scripts/overload_run.sh BACKEND OUTDIR   (e.g. onnx results/overload)
# Phases (same server, back to back): pre 48 users 30s -> spike 512 users 30s -> post1..3 48 users 15s each.
set -u
BACKEND=${1:?backend}; OUT=${2:?outdir}
cd ~/projects/ticketwise && source .venv/bin/activate
[ -e "$OUT" ] && { echo "$OUT exists; move it first" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
mkdir -p "$OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TICKETWISE_ONNX_THREADS=1 TICKETWISE_MODEL_BACKEND=$BACKEND \
TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 TICKETWISE_MAX_QUEUE_SIZE=64 \
  taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers 4 &
pid=$!
until curl -sf localhost:8000/health >/dev/null; do
  kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }; sleep 1
done
curl -s localhost:8000/health; echo
sleep 32
# memory sampler: total RSS (MB) of uvicorn parent + workers every 2 s
( while kill -0 $pid 2>/dev/null; do
    pids="$pid,$(pgrep -P $pid | paste -sd,)"
    echo "$(date +%s),$(ps -o rss= -p "$pids" 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s/1024}')" >> "$OUT/rss_mb.csv"
    sleep 2
  done ) &
sampler=$!
export TARGET=/search/semantic
run() {  # name users seconds
  taskset -c 4-7 locust -f scripts/locustfile_overload.py --headless -u $2 -r $2 -t ${3}s \
    --processes 3 --reset-stats --host http://127.0.0.1:8000 \
    --csv "$OUT/$1" --only-summary >/dev/null 2>&1
}
run warmup 48 15; rm -f "$OUT"/warmup_*
run pre 48 30
run spike 512 30
run post1 48 15
run post2 48 15
run post3 48 15
kill $sampler 2>/dev/null; kill $pid; sleep 5; pkill -f "uvicorn backend" 2>/dev/null
python - "$OUT" <<'PY'
import csv, sys
out = sys.argv[1]
print(f"{'phase':7} {'users':>5} {'accepted/s':>10} {'acc p50':>8} {'acc p99':>8} {'503/s':>8} {'other err':>9}")
users = {"pre": 48, "spike": 512, "post1": 48, "post2": 48, "post3": 48}
for ph in ["pre", "spike", "post1", "post2", "post3"]:
    rows = {r["Name"]: r for r in csv.DictReader(open(f"{out}/{ph}_stats.csv"))}
    a, j, e = rows.get("accepted"), rows.get("rejected_503"), rows.get("other_error")
    g = lambda r, k: float(r[k]) if r else 0.0
    print(f"{ph:7} {users[ph]:5d} {g(a,'Requests/s'):10.1f} {g(a,'50%'):8.0f} {g(a,'99%'):8.0f} "
          f"{g(j,'Requests/s'):8.1f} {int(g(e,'Request Count')):9d}")
rss = [int(l.split(",")[1]) for l in open(f"{out}/rss_mb.csv") if l.strip().split(",")[-1].isdigit()]
print(f"\nserver RSS (MB, all processes): start {rss[0]}, peak {max(rss)}, end {rss[-1]}")
PY
