import json

import httpx
import numpy as np

from backend.app.semantic_search import SemanticSearch


def test_triton_request_format_and_response_parsing():
    engine = SemanticSearch(backend="pytorch")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["path"] = request.url.path
        seen["body"] = body
        n = body["inputs"][0]["shape"][0]
        data = (np.ones((n, 384), dtype=np.float32) / np.sqrt(384)).ravel().tolist()
        return httpx.Response(200, json={"outputs": [{
            "name": "sentence_embedding", "shape": [n, 384],
            "datatype": "FP32", "data": data,
        }]})

    engine._triton = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://triton"
    )
    engine._triton_url = "/v2/models/minilm_batch/infer"
    engine._triton_inputs = ["input_ids", "attention_mask", "token_type_ids"]
    engine.triton_seq_len = 16

    embeddings = engine._encode_queries(["reset my password", "refund request"])

    assert seen["path"] == "/v2/models/minilm_batch/infer"
    assert [i["name"] for i in seen["body"]["inputs"]] == [
        "input_ids", "attention_mask", "token_type_ids",
    ]
    for item in seen["body"]["inputs"]:
        assert item["shape"] == [2, 16]
        assert item["datatype"] == "INT64"
        assert len(item["data"]) == 32
    assert embeddings.shape == (2, 384)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

    assert len(engine.search("reset my password", top_k=3)) == 3
