#!/usr/bin/env bash
# usage: bash scripts/interleave_triton.sh OUTDIR
# Interleaves ONNX in-process (C3/C4) and Triton (C5a/C5) at 64 users, 3 rounds, rotated order.
set -u
OUT=${1:?outdir}
cd ~/projects/ticketwise && source .venv/bin/activate
[ -e "$OUT" ] && { echo "$OUT exists; move it first" >&2; exit 1; }
curl -sf localhost:8000/health >/dev/null && { echo "Port 8000 in use" >&2; exit 1; }
docker rm -f triton >/dev/null 2>&1
curl -sf localhost:8100/v2/health/ready >/dev/null && { echo "Port 8100 in use" >&2; exit 1; }

cleanup() { pkill -f "uvicorn backend" 2>/dev/null; docker rm -f triton >/dev/null 2>&1; sleep 3; }

run_config() {  # $1 = config name, $2 = round
  local cfg=$1 r=$2 pid endpoints backend triton_model=minilm_batch
  case $cfg in
    onnx)     backend=onnx;   endpoints="/search/semantic /search/semantic/unbatched";;
    tbatch)   backend=triton; triton_model=minilm_batch;   endpoints="/search/semantic/unbatched";;
    tnobatch) backend=triton; triton_model=minilm_nobatch; endpoints="/search/semantic/unbatched";;
  esac
  if [ "$backend" = triton ]; then
    docker run -d --name triton --cpuset-cpus=0-3 -p 8100:8000 \
      -v "$PWD/triton_repo/model_repository:/models" \
      nvcr.io/nvidia/tritonserver:26.09-py3 tritonserver --model-repository=/models \
      --model-control-mode=explicit --load-model=$triton_model >/dev/null
    for i in $(seq 1 90); do curl -sf localhost:8100/v2/health/ready >/dev/null && break; sleep 2; done
    curl -sf localhost:8100/v2/health/ready >/dev/null || { echo "triton not ready" >&2; docker logs triton 2>&1 | tail -5; return 1; }
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TICKETWISE_ONNX_THREADS=1 TICKETWISE_MODEL_BACKEND=$backend \
  TICKETWISE_TRITON_MODEL=$triton_model TICKETWISE_TRITON_URL=http://127.0.0.1:8100 \
  TICKETWISE_MAX_BATCH_SIZE=32 TICKETWISE_MAX_WAIT_MS=0 \
    taskset -c 0-3 uvicorn backend.app.main:app --port 8000 --log-level warning --workers 4 &
  pid=$!
  until curl -sf localhost:8000/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server failed to start" >&2; return 1; }; sleep 1
  done
  curl -s localhost:8000/health | grep -q "\"model_backend\":\"$backend\"" || { echo "WRONG BACKEND" >&2; return 1; }
  sleep 32
  bash scripts/locust_run.sh "$OUT" ${cfg}_r$r "$endpoints" "64" 30 >/dev/null
  kill $pid; sleep 5; cleanup
}

read -ra CFGS <<< "${CONFIGS:-onnx tbatch tnobatch}"
ROUNDS=${ROUNDS:-3}
n=${#CFGS[@]}
for r in $(seq 1 $ROUNDS); do
  for k in $(seq 0 $((n-1))); do
    cfg=${CFGS[$(( (k + r - 1) % n ))]}
    echo "round $r: $cfg" >&2
    run_config $cfg $r || { cleanup; echo "FAILED $cfg round $r" >&2; }
  done
done
cleanup

OUT="$OUT" python - <<'PY'
import csv, glob, os, re, statistics as st
from collections import defaultdict
LABELS = {
    ("onnx", "_search_semantic"): "C4  ONNX + in-process batching",
    ("onnx", "_search_semantic_unbatched"): "C3  ONNX, no batching",
    ("tbatch", "_search_semantic_unbatched"): "C5  Triton, dynamic batching",
    ("tnobatch", "_search_semantic_unbatched"): "C5a Triton, no batching",
}
g = defaultdict(list)
for f in glob.glob(os.environ["OUT"] + "/*_stats.csv"):
    m = re.match(r"(onnx|tbatch|tnobatch)_r\d+(_search_semantic(?:_unbatched)?)_u64_stats\.csv", os.path.basename(f))
    if not m:
        continue
    for r in csv.DictReader(open(f)):
        if r["Name"] == "Aggregated":
            g[(m.group(1), m.group(2))].append(
                (float(r["Requests/s"]), float(r["95%"]), float(r["99%"]), float(r.get("Failure Count", 0) or 0)))
for key, label in LABELS.items():
    v = g.get(key)
    if not v:
        print(f"{label:32} no data")
        continue
    rps = [x[0] for x in v]
    print(f"{label:32} {st.median(rps):6.1f} [{min(rps):.0f}-{max(rps):.0f}]  p95 {st.median(x[1] for x in v):4.0f}  p99 {st.median(x[2] for x in v):4.0f}  failures {sum(x[3] for x in v):.0f}")
PY
