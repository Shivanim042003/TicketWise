import pytest

from backend.benchmark_api_backends import validate_semantic_response


def test_valid_semantic_response():
    validate_semantic_response({
        "method": "embedding",
        "results": [{"id": "TICK-001"}],
        "count": 1,
    })


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"method": "keyword", "results": [], "count": 0},
        {"method": "embedding", "results": {}, "count": 0},
        {"method": "embedding", "results": [], "count": 1},
        {"method": "embedding", "results": [{}], "count": True},
        {"method": "embedding", "results": [], "count": "0"},
    ],
)
def test_invalid_semantic_response_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_semantic_response(payload)
