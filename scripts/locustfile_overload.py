import json, os, random
from pathlib import Path
from locust import FastHttpUser, constant, task

TARGET = os.environ.get("TARGET", "/search/semantic")
raw = json.loads(Path("data/evaluation_queries.json").read_text())
items = raw if isinstance(raw, list) else raw.get("queries", [])
QUERIES = [q["query"] if isinstance(q, dict) else q for q in items]


class OverloadUser(FastHttpUser):
    wait_time = constant(0)

    @task
    def search(self):
        # 200 -> row "accepted", 503 -> row "rejected_503", anything else -> "other_error"
        with self.client.post(TARGET, json={"query": random.choice(QUERIES), "top_k": 3},
                              name="accepted", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code == 503:
                r.request_meta["name"] = "rejected_503"
                r.success()
            else:
                r.request_meta["name"] = "other_error"
                r.failure(f"status {r.status_code}")
