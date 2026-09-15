import json

import httpx

from legal_rag.embeddings import FakeEmbedder, OllamaEmbedder, slug


def _dot(x, y):
    return sum(p * q for p, q in zip(x, y))


def test_fake_embedder_is_deterministic_and_unit_norm():
    e = FakeEmbedder(dimension=32)
    a, b = e.embed_query("보증금 반환"), e.embed_query("보증금 반환")
    assert a == b and len(a) == 32
    assert abs(_dot(a, a) - 1.0) < 1e-6
    assert e.embed_query("불법행위 손해배상") != a
    assert e.name == "fake:32" and e.dimension == 32


def test_fake_embedder_shares_tokens_between_similar_texts():
    e = FakeEmbedder(dimension=32)
    q = e.embed_query("보증금 반환")
    assert _dot(q, e.embed_query("임대차 보증금 반환 청구")) > _dot(q, e.embed_query("교통사고 과실 비율"))


def test_ollama_embedder_batches_and_parses():
    calls = []

    def handler(request: httpx.Request):
        payload = json.loads(request.read())
        calls.append(payload)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in payload["input"]]})

    e = OllamaEmbedder(model="bge-m3", base_url="http://test", batch_size=2, transport=httpx.MockTransport(handler))
    vectors = e.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and len(calls) == 2 and calls[0]["model"] == "bge-m3"
    assert e.name == "ollama:bge-m3" and e.dimension == 2


def test_slug():
    assert slug("ollama:bge-m3") == "ollama_bge_m3" and slug("fake:64") == "fake_64"


def test_ollama_refuses_silent_truncation():
    def handler(request):
        payload = json.loads(request.read())
        assert payload['truncate'] is False
        return httpx.Response(200, json={'embeddings': [[0.1, 0.2]]})

    embedder = OllamaEmbedder(model='bge-m3', base_url='http://test', transport=httpx.MockTransport(handler))
    assert embedder.embed_query('원문 전체') == [0.1, 0.2]
