#!/usr/bin/env bash
# runs workers:threads configs; batched and unbatched; Locust at several user counts
set -u
cd ~/projects/ticketwise && source .venv/bin/activate
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use: stop the server first" >&2; exit 1; }
for cfg in 1:4 2:2 4:1; do
  w=${cfg%:*}; t=${cfg#*:}
  OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 \
    taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers $w &
  pid=$!
  until curl -sf localhost:8000/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; exit 1; }
    sleep 1
  done
  sleep $((w * 8))   # let every worker finish loading the model
  echo "=== workers=$w threads=$t ==="
  bash scripts/locust_run.sh results/baseline w${w}t${t} "/search/semantic /search/semantic/unbatched" "8 16 32 64" 30
  kill $pid; sleep 5; pkill -f "uvicorn backend" 2>/dev/null; sleep 10
done
