#!/usr/bin/env bash
# usage: scripts/sweep.sh OUTDIR size:wait [size:wait ...]   (CONC="1 5 10" overrides concurrency)
set -u
out=$1; shift
cd ~/projects/ticketwise && source .venv/bin/activate
avail=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
[ "$avail" -ge 10 ] || { echo "C: has only ${avail} GB free; need 10" >&2; exit 1; }
if curl -sf localhost:8000/health >/dev/null; then
  echo "Port 8000 already in use: stop the running server first." >&2; exit 1
fi
mkdir -p "$out"
for cfg in "$@"; do
  size=${cfg%:*}; w=${cfg#*:}
  TICKETWISE_MAX_BATCH_SIZE=$size TICKETWISE_MAX_WAIT_MS=$w \
    taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning &
  pid=$!
  until curl -sf localhost:8000/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }
    sleep 1
  done
  for c in ${CONC:-1 5 10 20}; do for r in 1 2 3; do
    taskset -c 4-5 python -m backend.benchmark --requests 1000 --concurrency $c \
      --output "$out/size${size}_wait${w}_c${c}_r${r}.csv" >/dev/null || exit 1
  done; done
  echo "=== max_batch_size=$size max_wait_ms=$w ==="
  python -m backend.summarize_csv "$out"/size${size}_wait${w}_*.csv
  curl -s localhost:8000/metrics/batching; echo
  kill $pid; wait $pid 2>/dev/null
  sleep 30
done
