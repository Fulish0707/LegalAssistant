"""Contracts for the two latency changes; no external model or index is used."""
from unittest.mock import Mock

import pytest

from legal_rag.llm import FakeChatModel
from legal_rag.llm.prompts import load_prompt
from legal_rag.rag.generator import Generator
from legal_rag.retrieval import DenseRetriever, HybridRetriever
from legal_rag.schemas import LegalAnswer, RetrievalFilters
from tests.legal_rag.unit.test_verification import _answer, _bundle


def hybrid(embedders, *, dense_class=DenseRetriever):
    legs = []
    for i, embedder in enumerate(embedders):
        store = Mock()
        store.search_dense.return_value = []
        leg = dense_class(store, embedder)
        leg.name = f"corpus-{i}:dense"
        legs.append(leg)
    sparse = Mock(name="custom")
    sparse.name = "sparse"
    sparse.retrieve.return_value = []
    fuser, reranker = Mock(), Mock()
    fuser.fuse.return_value = []
    reranker.rerank.return_value = []
    return HybridRetriever(legs + [sparse], fuser, reranker, k_leg=100, k_fused=100)


def embedder():
    value = Mock()
    value.embed_query.side_effect = lambda q: [float(len(q)), 1.0]
    return value


def test_five_corpora_share_one_embedding_but_keep_all_ranked_lists():
    model = embedder()
    retriever = hybrid([model] * 5)
    filters = RetrievalFilters()
    retriever.retrieve_many(["query", "another", "query"], 5, filters, rerank_query="original")
    assert model.embed_query.call_count == 2
    for leg in retriever.legs[:5]:
        assert leg.store.search_dense.call_count == 3
        assert all(call.args[1:] == (100, filters) for call in leg.store.search_dense.call_args_list)
    assert len(retriever.fuser.fuse.call_args.args[0]) == 18
    assert retriever.legs[-1].retrieve.call_count == 3
    retriever.reranker.rerank.assert_called_once_with("original", [], 5)


def test_cache_does_not_cross_requests_or_embedder_instances():
    a, b = embedder(), embedder()
    retriever = hybrid([a, a, b])
    retriever.retrieve_many(["same"], 5)
    retriever.retrieve_many(["same"], 5)
    assert a.embed_query.call_count == b.embed_query.call_count == 2


def test_dense_subclass_keeps_its_override():
    class CustomDense(DenseRetriever):
        def retrieve(self, query, k, filters=None):
            self.seen = (query, k, filters)
            return []
    model = embedder()
    retriever = hybrid([model], dense_class=CustomDense)
    retriever.retrieve_many(["q"], 5)
    assert retriever.legs[0].seen == ("q", 100, None)
    model.embed_query.assert_not_called()


@pytest.mark.parametrize("style", ["full", "concise"])
def test_style_keeps_schema_budget_and_evidence(style):
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    Generator(model, answer_style=style, max_tokens=2048).generate("q", _bundle())
    call = model.calls[0]
    assert call.response_schema is LegalAnswer and call.max_tokens == 2048
    expected = load_prompt("answer_system")
    if style == "concise":
        expected += "\n\n" + load_prompt("answer_concise")
    assert call.messages[0].content == expected
    assert "[LEGAL SOURCE S1]" in call.messages[1].content


def test_unknown_style_fails_instead_of_silently_changing_prompt():
    with pytest.raises(ValueError, match="unknown answer style"):
        Generator(FakeChatModel({}), answer_style="unknown")
