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


def _recording_embedder(**kwargs):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})

    embedder = OllamaEmbedder(model="bge-m3", base_url="http://test", transport=httpx.MockTransport(handler), **kwargs)
    return embedder, bodies


def test_query_context_window_is_sent_only_with_queries():
    """bge-m3 at Ollama's default 4096 context reserves a 1.17 GB compute buffer, which
    did not fit beside qwen3:4b at 8192 and the reranker (2026-09-27). Queries are short;
    index builds keep the full window, and truncate=False still refuses silent cuts."""
    embedder, bodies = _recording_embedder(query_num_ctx=1024)
    embedder.embed_query("보증금 반환")
    embedder.embed_documents(["긴 판결문"])

    assert bodies[0]["options"] == {"num_ctx": 1024}
    assert "options" not in bodies[1]
    assert bodies[0]["truncate"] is False


def test_no_query_context_window_by_default():
    embedder, bodies = _recording_embedder()
    embedder.embed_query("q")
    assert "options" not in bodies[0]


def test_factory_passes_the_query_context_window():
    from legal_rag.config import Settings
    from legal_rag.factory import build_embedder

    embedder = build_embedder(Settings(embedding={"query_num_ctx": 1024}))
    assert embedder.query_num_ctx == 1024
    assert Settings().embedding.query_num_ctx is None


def test_query_embeddings_can_run_on_cpu():
    """num_gpu 0 frees the GPU for a larger answer model: EXAONE 3.5 7.8B at 12288 context and the fp16
    reranker peaked at 7,642 of 8,188 MiB with bge-m3 on CPU, and CPU query vectors matched GPU ones
    (cosine >= 0.99993 on 50 questions, spike 005, 2026-11-23). Document embeddings keep Ollama's default."""
    embedder, bodies = _recording_embedder(query_num_ctx=1024, query_num_gpu=0)
    embedder.embed_query("보증금 반환")
    embedder.embed_documents(["긴 판결문"])
    assert bodies[0]["options"] == {"num_ctx": 1024, "num_gpu": 0}
    assert "options" not in bodies[1]


def test_query_num_gpu_alone_is_sent():
    embedder, bodies = _recording_embedder(query_num_gpu=0)
    embedder.embed_query("q")
    assert bodies[0]["options"] == {"num_gpu": 0}


def test_factory_passes_query_num_gpu():
    from legal_rag.config import Settings
    from legal_rag.factory import build_embedder

    assert build_embedder(Settings(embedding={"query_num_gpu": 0})).query_num_gpu == 0
    assert Settings().embedding.query_num_gpu is None
