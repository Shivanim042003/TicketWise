#!/usr/bin/env bash
# usage: scripts/locust_run.sh OUTDIR TAG "ENDPOINTS" "USERS" [SECONDS]
set -u
out=$1; tag=$2; eps=$3; users=$4; secs=${5:-45}
cd ~/projects/ticketwise && source .venv/bin/activate
mkdir -p "$out"
curl -sf localhost:8000/health >/dev/null || { echo "server not running" >&2; exit 1; }
maxu=$(echo $users | tr ' ' '\n' | sort -n | tail -1)
for t in $eps; do   # warm-up, not recorded
  TARGET=$t taskset -c 4-7 locust -f scripts/locustfile.py --headless -u $maxu -r $maxu -t 15s \
    --processes 3 --host http://127.0.0.1:8000 --only-summary >/dev/null 2>&1
done
for t in $eps; do for u in $users; do
  name=$(echo $t | tr '/' '_')
  TARGET=$t taskset -c 4-7 locust -f scripts/locustfile.py --headless -u $u -r $u -t ${secs}s \
    --processes 3 --reset-stats --host http://127.0.0.1:8000 \
    --csv "$out/${tag}${name}_u${u}" --only-summary >/dev/null 2>&1
done; done
python - "$out" "$tag" <<'PY'
import csv, glob, sys
out, tag = sys.argv[1:3]
print(f"{'file':46} {'reqs':>7} {'fail':>5} {'rps':>8} {'p50':>6} {'p95':>6} {'p99':>6}")
for f in sorted(glob.glob(f"{out}/{tag}_*_stats.csv")):
    for r in csv.DictReader(open(f)):
        if r["Name"] == "Aggregated":
            print(f"{f.split('/')[-1]:46} {r['Request Count']:>7} {r['Failure Count']:>5} {float(r['Requests/s']):8.1f} {r['50%']:>6} {r['95%']:>6} {r['99%']:>6}")
PY
