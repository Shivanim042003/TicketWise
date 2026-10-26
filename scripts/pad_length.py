import json
import math
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
data = json.load(open("data/evaluation_queries.json", encoding="utf-8"))
items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
queries = [q if isinstance(q, str) else (q.get("query") or q.get("text") or "") for q in items]
lengths = sorted(len(tok(q)["input_ids"]) for q in queries if q)
longest = lengths[-1]
pad = min(64, max(16, math.ceil(longest / 8) * 8))
print(f"{len(lengths)} queries, token lengths min {lengths[0]} median {lengths[len(lengths)//2]} max {longest} -> pad {pad}", file=sys.stderr)
print(pad)
