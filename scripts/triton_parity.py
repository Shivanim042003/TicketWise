from backend.app.semantic_search import SemanticSearch

QUERIES = ["reset my password", "refund for double charge", "app crashes on login",
           "cannot export report to csv", "invoice shows wrong tax", "API returns 503 after deployment"]

pt = SemanticSearch(backend="pytorch")
tr = SemanticSearch(backend="triton")
a, b = pt._encode_queries(QUERIES), tr._encode_queries(QUERIES)
print("shape", a.shape, b.shape, "| min cosine vs PyTorch:", round(float((a * b).sum(axis=1).min()), 6))
print("top-1 ticket agrees:", [pt.search(q, 1)[0]["id"] == tr.search(q, 1)[0]["id"] for q in QUERIES])
