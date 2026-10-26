"""Held-out retrieval evaluation: BM25 vs MiniLM vs bge-small.

Splits data/evaluation_queries_v4.json into dev/test (stratified by type, fixed seed).
The no-match threshold is tuned on dev only and applied to test.
Usage: python scripts/eval_extended.py [--models bm25,minilm,bge] [--seed 13]
"""
import argparse, json, math, random, re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-small-en-v1.5",
            "Represent this sentence for searching relevant passages: "),
}


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def doc_text(t):
    return f"{t['title']}. {t['description']}"


def split(queries, seed, test_frac=0.5):
    by_type = defaultdict(list)
    for i, q in enumerate(queries):
        by_type[q["type"]].append(i)
    dev, test = [], []
    for _, idx in sorted(by_type.items()):
        random.Random(seed).shuffle(idx)
        n_test = round(len(idx) * test_frac)
        test += idx[:n_test]
        dev += idx[n_test:]
    return sorted(dev), sorted(test)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / len(self.docs)
        df = defaultdict(int)
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        q = tok(query)
        for i, d in enumerate(self.docs):
            tf = defaultdict(int)
            for w in d:
                tf[w] += 1
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out[i] = s
        return out


def make_scorer(name, docs):
    if name == "bm25":
        bm = BM25(docs)
        return bm.scores
    from sentence_transformers import SentenceTransformer
    model_id, prefix = MODELS[name]
    model = SentenceTransformer(model_id)
    d = model.encode(docs, normalize_embeddings=True)

    def score(query):
        q = model.encode([prefix + query], normalize_embeddings=True)[0]
        return d @ q
    return score


def metrics(scores, ids, relevant):
    order = [ids[i] for i in np.argsort(-scores)]
    rel = set(relevant)
    r = {}
    for k in (1, 3, 5):
        r[f"R@{k}"] = len(rel & set(order[:k])) / len(rel)
    rr = 0.0
    for rank, t in enumerate(order[:10], 1):
        if t in rel:
            rr = 1.0 / rank
            break
    r["MRR@10"] = rr
    dcg = sum(1.0 / math.log2(rank + 1) for rank, t in enumerate(order[:10], 1) if t in rel)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), 10) + 1))
    r["nDCG@10"] = dcg / idcg
    return r


def best_threshold(top, has_rel):
    cands = sorted(set(top))
    cands = [(a + b) / 2 for a, b in zip(cands, cands[1:])] or cands
    best_t, best_acc = cands[0], -1
    for t in cands:
        acc = np.mean([(s >= t) == h for s, h in zip(top, has_rel)])
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def run(name, tickets, queries, dev, test):
    ids = [t["id"] for t in tickets]
    score = make_scorer(name, [doc_text(t) for t in tickets])
    S = [score(q["query"]) for q in queries]
    top = [float(s.max()) for s in S]
    has_rel = [bool(q["relevant_ticket_ids"]) for q in queries]
    thr, dev_acc = best_threshold([top[i] for i in dev], [has_rel[i] for i in dev])
    test_acc = float(np.mean([(top[i] >= thr) == has_rel[i] for i in test]))
    res = {}
    for split_name, idx in (("dev", dev), ("test", test)):
        rows = [metrics(S[i], ids, queries[i]["relevant_ticket_ids"]) for i in idx if has_rel[i]]
        res[split_name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res[split_name]["n"] = len(rows)
    res["threshold"] = float(thr)
    res["reject_acc_dev"] = float(dev_acc)
    res["reject_acc_test"] = test_acc
    res["test_failures"] = [
        {"query": queries[i]["query"], "type": queries[i]["type"],
         "relevant": queries[i]["relevant_ticket_ids"],
         "top3": [ids[j] for j in np.argsort(-S[i])[:3]]}
        for i in test if has_rel[i]
        and ids[int(np.argmax(S[i]))] not in queries[i]["relevant_ticket_ids"]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bm25,minilm,bge")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--queries", default="data/evaluation_queries_v4.json")
    a = ap.parse_args()
    tickets = json.load(open(ROOT / "data/tickets.json", encoding="utf-8"))
    queries = json.load(open(ROOT / a.queries, encoding="utf-8"))
    dev, test = split(queries, a.seed)
    print(f"tickets={len(tickets)} queries={len(queries)} dev={len(dev)} test={len(test)}\n")
    out = {}
    cols = ["R@1", "R@3", "R@5", "MRR@10", "nDCG@10"]
    print("| Model | n | " + " | ".join(cols) + " | No-match acc (test) |")
    print("|---|---:|" + "---:|" * (len(cols) + 1))
    for name in a.models.split(","):
        out[name] = run(name, tickets, queries, dev, test)
        t = out[name]["test"]
        print(f"| {name} | {t['n']} | " + " | ".join(f"{t[c]:.3f}" for c in cols)
              + f" | {out[name]['reject_acc_test']:.3f} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/retrieval_extended.json").write_text(json.dumps(out, indent=2))
    print("\nTest-set failures (top-1 not relevant):")
    for name, r in out.items():
        print(f"- {name}: {len(r['test_failures'])}")
    print("saved results/retrieval_extended.json")


if __name__ == "__main__":
    main()
