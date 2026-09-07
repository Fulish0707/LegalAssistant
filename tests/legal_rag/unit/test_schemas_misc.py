from legal_rag.schemas import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    Evidence,
    EvidenceBundle,
    LegalAnswer,
    QueryAnalysis,
    RetrievalExample,
    RetrievalFilters,
    ScoredChunk,
)


def _chunk(cid="statute:abc#art:제750조"):
    return Chunk(
        id=cid, document_id="statute:abc", document_type=DocumentType.statute, path="art:제750조", order=0,
        text="민법 제750조(불법행위의 내용)\n고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        title="민법", metadata=ChunkMetadata(law_name="민법", article="제750조", dataset="fixture", corpus_version="v1"),
    )


def test_chunk_metadata_defaults_are_none():
    m = _chunk().metadata
    assert m.court is None and m.paragraph is None and m.effective_date is None


def test_scored_chunk_carries_retriever_name():
    s = ScoredChunk(chunk=_chunk(), score=0.5, retriever="dense")
    assert s.retriever == "dense"


def test_evidence_bundle_renders_source_blocks():
    bundle = EvidenceBundle(items=[
        Evidence(source_id="S1", chunk_id=_chunk().id, document_id="statute:abc", document_type=DocumentType.statute,
                 header={"Type": "Statute", "Law": "민법", "Article": "제750조"}, text="고의 또는 과실로...", score=0.9)
    ])
    rendered = bundle.render()
    assert rendered.startswith("[LEGAL SOURCE S1]")
    assert "Law: 민법" in rendered and "Article: 제750조" in rendered and "Source ID: S1" in rendered
    assert bundle.by_id()["S1"].chunk_id == _chunk().id


def test_legal_answer_json_schema_has_required_fields():
    schema = LegalAnswer.model_json_schema()
    for field in ("summary", "analysis", "confidence"):
        assert field in schema["required"]


def test_query_analysis_defaults():
    qa = QueryAnalysis()
    assert qa.needs_more_info is False and qa.keywords == [] and qa.document_types_needed == []


def test_retrieval_filters_is_empty():
    assert RetrievalFilters().is_empty() and not RetrievalFilters(legal_domain="민사").is_empty()


def test_retrieval_example_parses_filters():
    ex = RetrievalExample(id="q1", query="보증금", relevant_document_ids=["statute:abc"],
                          filters=RetrievalFilters(document_types=[DocumentType.statute]))
    assert ex.filters.document_types == [DocumentType.statute]
