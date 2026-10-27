import json, os, random
from pathlib import Path
import gevent
from locust import FastHttpUser, constant, task

TARGET = os.environ.get("TARGET", "/search/semantic")
HONOR_RETRY = os.environ.get("HONOR_RETRY", "0") == "1"   # sleep 1 s (the Retry-After value) after a 503
SLO_MS = float(os.environ.get("SLO_MS", "200"))
raw = json.loads(Path("data/evaluation_queries.json").read_text())
items = raw if isinstance(raw, list) else raw.get("queries", [])
QUERIES = [q["query"] if isinstance(q, dict) else q for q in items]


class OverloadUser(FastHttpUser):
    wait_time = constant(0)

    @task
    def search(self):
        # 200 -> row "accepted" (+ row "within_slo" if fast enough), 503 -> "rejected_503", else "other_error"
        rejected = False
        with self.client.post(TARGET, json={"query": random.choice(QUERIES), "top_k": 3},
                              name="accepted", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
                if r.request_meta.get("response_time", 1e9) <= SLO_MS:
                    self.environment.events.request.fire(
                        request_type="POST", name="within_slo",
                        response_time=r.request_meta["response_time"], response_length=0,
                        exception=None, context={})
            elif r.status_code == 503:
                r.request_meta["name"] = "rejected_503"
                r.success()
                rejected = True
            else:
                r.request_meta["name"] = "other_error"
                r.failure(f"status {r.status_code}")
        if rejected and HONOR_RETRY:
            gevent.sleep(1.0)
