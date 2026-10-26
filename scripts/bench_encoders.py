"""Raw ONNX encode cost: all-MiniLM-L6-v2 vs bge-small-en-v1.5.

One intra-op thread (matches the 4 workers x 1 thread setup), real queries from
data/evaluation_queries_v4.json, batches padded to the longest query in the batch.
Models alternate each round to limit machine drift.
Usage: python scripts/bench_encoders.py [--rounds 5]
"""
import argparse, json, statistics, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}
BATCHES = (1, 8, 32)


def load(repo):
    path = hf_hub_download(repo, "onnx/model.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return AutoTokenizer.from_pretrained(repo), ort.InferenceSession(
        path, opts, providers=["CPUExecutionProvider"])


def run_batch(tok, sess, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
    feed = {i.name: enc[i.name].astype(np.int64) for i in sess.get_inputs() if i.name in enc}
    t = time.perf_counter()
    sess.run(None, feed)
    return time.perf_counter() - t, int(enc["input_ids"].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    queries = [q["query"] for q in json.load(open(ROOT / "data/evaluation_queries_v4.json"))]
    loaded = {n: load(r) for n, r in MODELS.items()}
    res = {n: {b: [] for b in BATCHES} for n in MODELS}
    for rnd in range(a.rounds):
        order = list(MODELS) if rnd % 2 == 0 else list(MODELS)[::-1]
        for n in order:
            tok, sess = loaded[n]
            for b in BATCHES:
                for i in range(5):  # warm-up
                    run_batch(tok, sess, [queries[(i * b + j) % len(queries)] for j in range(b)])
                times = []
                for i in range(60):
                    texts = [queries[(i * b + j) % len(queries)] for j in range(b)]
                    times.append(run_batch(tok, sess, texts)[0])
                res[n][b].append(statistics.median(times))
    print("| Batch | MiniLM ms/batch | bge ms/batch | MiniLM q/s/core | bge q/s/core | bge cost vs MiniLM |")
    print("|---:|---:|---:|---:|---:|---:|")
    out = {}
    for b in BATCHES:
        m = statistics.median(res["minilm"][b]); g = statistics.median(res["bge"][b])
        out[b] = {"minilm_ms": m * 1e3, "bge_ms": g * 1e3, "ratio": g / m}
        print(f"| {b} | {m*1e3:.1f} | {g*1e3:.1f} | {b/m:.0f} | {b/g:.0f} | {g/m:.2f}x |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/encoder_cost.json").write_text(json.dumps(out, indent=2))
    print("saved results/encoder_cost.json")


if __name__ == "__main__":
    main()
