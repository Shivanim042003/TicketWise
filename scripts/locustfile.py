import json, os, random
from pathlib import Path
from locust import FastHttpUser, constant, task

TARGET = os.environ.get("TARGET", "/search/semantic")
raw = json.loads(Path("data/evaluation_queries.json").read_text())
items = raw if isinstance(raw, list) else raw.get("queries", [])
QUERIES = [q["query"] if isinstance(q, dict) else q for q in items]

class SearchUser(FastHttpUser):
    wait_time = constant(0)  # closed loop: next request as soon as the previous returns

    @task
    def search(self):
        self.client.post(TARGET, json={"query": random.choice(QUERIES), "top_k": 3}, name=TARGET)
