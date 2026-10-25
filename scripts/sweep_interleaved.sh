#!/usr/bin/env bash
# usage: ROUNDS=3 CONC="1 10 20 40" scripts/sweep_interleaved.sh OUTDIR size:wait [size:wait ...]
# Each round runs every config once, rotating the order, so machine drift averages out.
set -u
out=$1; shift
cfgs=("$@"); n=${#cfgs[@]}
cd ~/projects/ticketwise && source .venv/bin/activate
avail=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
[ "$avail" -ge 10 ] || { echo "C: has only ${avail} GB free; need 10" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
mkdir -p "$out"
for r in $(seq 1 ${ROUNDS:-3}); do
  for k in $(seq 0 $((n-1))); do
    cfg=${cfgs[$(( (k + r - 1) % n ))]}
    size=${cfg%:*}; w=${cfg#*:}
    TICKETWISE_MAX_BATCH_SIZE=$size TICKETWISE_MAX_WAIT_MS=$w \
      taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning &
    pid=$!
    until curl -sf localhost:8000/health >/dev/null; do
      kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }
      sleep 1
    done
    for c in ${CONC:-1 10 20 40}; do
      taskset -c 4-5 python -m backend.benchmark --requests 1000 --concurrency $c \
        --output "$out/size${size}_wait${w}_c${c}_r${r}.csv" >/dev/null || exit 1
    done
    echo "round $r size=$size wait=$w $(curl -s localhost:8000/metrics/batching)"
    kill $pid; wait $pid 2>/dev/null
    sleep 20
  done
done
for cfg in "${cfgs[@]}"; do
  size=${cfg%:*}; w=${cfg#*:}
  echo "=== max_batch_size=$size max_wait_ms=$w ==="
  python -m backend.summarize_csv "$out"/size${size}_wait${w}_*.csv
done
