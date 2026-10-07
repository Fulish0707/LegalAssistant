from datetime import date

from legal_rag.llm import FakeChatModel
from legal_rag.rag.context import ContextBuilder
from legal_rag.rag.rewriter import QueryRewriter
from legal_rag.retrieval.filters import to_conditions
from legal_rag.schemas import Chunk, ChunkMetadata, CitationKind, DocumentType, QueryAnalysis, RewrittenQueries, ScoredChunk
from tests.legal_rag.unit.test_store import _chunks


def test_deterministic_rewrite_adds_refs_and_keywords():
    analysis = QueryAnalysis(issues=["보증금 반환"], keywords=["임대차", "보증금"], statute_refs=["주택임대차보호법 제3조의3"])
    rq = QueryRewriter(model=None).rewrite("집주인이 보증금을 안 줍니다", analysis)
    assert rq.search_queries[0] == "집주인이 보증금을 안 줍니다"
    assert "주택임대차보호법 제3조의3" in rq.search_queries
    assert "보증금 반환 임대차 보증금" in rq.search_queries
    assert len(rq.search_queries) <= 4


def test_llm_rewrite_appends_model_queries_and_survives_failure():
    model = FakeChatModel({"RewrittenQueries": [RewrittenQueries(original="q", search_queries=["임대차 종료 보증금반환의무"])]})
    rq = QueryRewriter(model=model).rewrite("보증금", QueryAnalysis())
    assert rq.search_queries == ["보증금", "임대차 종료 보증금반환의무"]
    assert QueryRewriter(model=FakeChatModel({})).rewrite("보증금", QueryAnalysis()).search_queries == ["보증금"]


def test_filters_from_analysis():
    f = QueryRewriter.filters(QueryAnalysis(domain="민사", document_types_needed=[DocumentType.court_decision]))
    assert f.citation_kind is CitationKind.case and f.document_types is None and f.legal_domain is None
    assert QueryRewriter.filters(QueryAnalysis(domain="민사")) is None
    two = QueryRewriter.filters(QueryAnalysis(document_types_needed=[DocumentType.statute, DocumentType.court_decision]))
    assert two is None
    assert QueryRewriter.filters(QueryAnalysis()) is None


def test_case_filter_admits_the_imported_case_types():
    """The analyzer names the native court_decision type, but the imported corpora store
    cases as judgment, decision, constitutional_decision and patent_trial_decision. An exact
    document_type filter matched nothing and /v1/chat answered with no evidence (2026-09-27)."""
    f = QueryRewriter.filters(QueryAnalysis(document_types_needed=[DocumentType.court_decision]))
    (condition,) = to_conditions(f)
    assert {"judgment", "decision", "constitutional_decision", "patent_trial_decision"} <= set(condition.values)


def test_several_types_of_one_kind_still_filter_by_that_kind():
    f = QueryRewriter.filters(QueryAnalysis(document_types_needed=[DocumentType.judgment, DocumentType.court_decision]))
    assert f.citation_kind is CitationKind.case


def test_context_builder_assigns_ids_and_headers():
    c1, c2, c3 = _chunks()
    hits = [ScoredChunk(chunk=c3, score=0.9, retriever="reranked"), ScoredChunk(chunk=c1, score=0.8, retriever="reranked")]
    bundle = ContextBuilder(max_chars=1500).build(hits)
    assert [e.source_id for e in bundle.items] == ["S1", "S2"]
    assert bundle.items[0].header["Type"] == "Case" and bundle.items[0].header["Case number"] == "2018다12345"
    assert bundle.items[0].header["Decision date"] == "2019-05-30"
    assert bundle.items[1].header["Law"] == "주택임대차보호법" and bundle.items[1].header["Article"] == "제3조의2"
    assert "Paragraph" not in bundle.items[1].header
    assert "[LEGAL SOURCE S1]" in bundle.render()


def test_context_builder_truncates_long_text():
    c1, *_ = _chunks()
    long = c1.model_copy(update={"text": "가" * 5000})
    bundle = ContextBuilder(max_chars=100).build([ScoredChunk(chunk=long, score=1, retriever="reranked")])
    assert len(bundle.items[0].text) == 101 and bundle.items[0].text.endswith("…")


def test_context_builder_renders_imported_judgment_as_case():
    # An imported judgment carries the same court/case metadata as a native
    # court_decision and must not fall through to the interpretation header.
    chunk = Chunk(
        id="d9#p", document_id="d9", document_type=DocumentType.judgment, path="p", order=0,
        text="피고인은 무죄", title="t",
        metadata=ChunkMetadata(dataset="f", corpus_version="v", court="인천지방법원",
                               case_number="2008보5", decision_date=date(2008, 7, 14)),
    )
    bundle = ContextBuilder().build([ScoredChunk(chunk=chunk, score=1.0, retriever="reranked")])
    header = bundle.items[0].header
    assert header["Type"] == "Case"
    assert header["Court"] == "인천지방법원" and header["Case number"] == "2008보5"
    assert "Authority" not in header


def test_analyzer_terms_can_be_kept_out_of_the_search():
    # The analyzer's issues and keywords differ from run to run, and with them the evidence:
    # two runs of the same 80 questions shared identical evidence on 47 to 50 (2026-10-05).
    analysis = QueryAnalysis(issues=["보증금 반환"], keywords=["임대차", "보증금"], statute_refs=["주택임대차보호법 제3조의3"])
    rq = QueryRewriter(model=None, analyzer_terms=False).rewrite("집주인이 보증금을 안 줍니다", analysis)
    assert rq.search_queries == ["집주인이 보증금을 안 줍니다", "주택임대차보호법 제3조의3"]
