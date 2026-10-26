"""Retrieval evaluation on a public BEIR dataset (default: SciFact).

Downloads the benchmark zip once to ~/.cache/ticketwise_beir, then compares
BM25 vs MiniLM vs bge-small on the official test queries (labels by others).
Usage: python scripts/eval_public.py [--dataset scifact] [--models bm25,minilm,bge]
"""
import argparse, csv, io, json, math, re, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "ticketwise_beir"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-small-en-v1.5",
            "Represent this sentence for searching relevant passages: "),
}


def load(name):
    folder = CACHE / name
    if not (folder / "corpus.jsonl").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        data = urllib.request.urlopen(URL.format(name), timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(CACHE)
    corpus = [json.loads(l) for l in open(folder / "corpus.jsonl", encoding="utf-8")]
    queries = {}
    for l in open(folder / "queries.jsonl", encoding="utf-8"):
        d = json.loads(l)
        queries[d["_id"]] = d["text"]
    qrels = defaultdict(dict)
    with open(folder / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return corpus, queries, qrels


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        toks = [tok(d) for d in docs]
        self.n = len(toks)
        self.lens = np.array([len(t) for t in toks], dtype=float)
        self.avg = self.lens.mean()
        self.post = defaultdict(list)
        for i, t in enumerate(toks):
            tf = defaultdict(int)
            for w in t:
                tf[w] += 1
            for w, f in tf.items():
                self.post[w].append((i, f))
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in self.post.items()}

    def scores(self, query):
        out = np.zeros(self.n)
        for w in tok(query):
            for i, f in self.post.get(w, ()):
                out[i] += self.idf[w] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out


def make_scorer(name, docs):
    if name == "bm25":
        return BM25(docs).scores
    from sentence_transformers import SentenceTransformer
    model_id, prefix = MODELS[name]
    model = SentenceTransformer(model_id)
    model.max_seq_length = 256
    d = model.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    def score(query):
        return d @ model.encode([prefix + query], normalize_embeddings=True)[0]
    return score


def evaluate(scorer, ids, queries, qrels):
    r5, r10, mrr, ndcg = [], [], [], []
    for qid, rel in qrels.items():
        if qid not in queries:
            continue
        s = scorer(queries[qid])
        top = np.argpartition(-s, 10)[:10]
        order = [ids[i] for i in top[np.argsort(-s[top])]]
        relset = set(rel)
        r5.append(len(relset & set(order[:5])) / len(relset))
        r10.append(len(relset & set(order[:10])) / len(relset))
        rr = next((1.0 / k for k, t in enumerate(order, 1) if t in relset), 0.0)
        mrr.append(rr)
        dcg = sum(rel.get(t, 0) / math.log2(k + 1) for k, t in enumerate(order, 1))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(k + 1) for k, g in enumerate(ideal, 1))
        ndcg.append(dcg / idcg)
    return {"n": len(ndcg), "R@5": float(np.mean(r5)), "R@10": float(np.mean(r10)),
            "MRR@10": float(np.mean(mrr)), "nDCG@10": float(np.mean(ndcg))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--models", default="bm25,minilm,bge")
    a = ap.parse_args()
    corpus, queries, qrels = load(a.dataset)
    ids = [d["_id"] for d in corpus]
    docs = [(d.get("title", "") + ". " + d.get("text", "")).strip() for d in corpus]
    print(f"{a.dataset}: docs={len(docs)} test queries={len(qrels)}\n")
    cols = ["R@5", "R@10", "MRR@10", "nDCG@10"]
    print("| Model | n | " + " | ".join(cols) + " |")
    print("|---|---:|" + "---:|" * len(cols))
    out = {}
    for name in a.models.split(","):
        out[name] = evaluate(make_scorer(name, docs), ids, queries, qrels)
        r = out[name]
        print(f"| {name} | {r['n']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/retrieval_public_{a.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/retrieval_public_{a.dataset}.json")


if __name__ == "__main__":
    main()
