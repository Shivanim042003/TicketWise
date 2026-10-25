import time
import numpy as np
from backend.app.semantic_search import SemanticSearch

QUERIES = ["reset my password", "refund for double charge", "app crashes on login",
           "cannot export report to csv", "invoice shows wrong tax", "how do I delete my account"]

t0 = time.time(); pt = SemanticSearch(backend="pytorch"); print(f"pytorch load {time.time()-t0:.1f}s")
t0 = time.time(); ox = SemanticSearch(backend="onnx");    print(f"onnx load    {time.time()-t0:.1f}s")

a = pt.model.encode(QUERIES, normalize_embeddings=True)
b = ox.model.encode(QUERIES, normalize_embeddings=True)
cos = (a * b).sum(axis=1)
print("shape", a.shape, b.shape)
print("min cosine", float(cos.min()), "max abs diff", float(np.abs(a - b).max()))
