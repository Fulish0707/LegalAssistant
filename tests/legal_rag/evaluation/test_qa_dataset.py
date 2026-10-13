import json

import pytest

from legal_rag.evaluation.qa_dataset import build_retrieval_examples, corpus_document_ids


def write_qa(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def qa_row(**kw):
    base = dict(document_id="criminal::judgment::64516", source_key="HS_P_64516",
                legal_domain="criminal", document_type="judgment",
                question="보증금을 돌려받을 수 있나요?", answer="가능합니다. 임차권등기명령을 신청하십시오.",
                metadata={"source": "AIHub"})
    base.update(kw)
    return base


def test_question_and_document_id_become_a_retrieval_example(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row()])
    examples, report = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert len(examples) == 1
    assert examples[0].query == "보증금을 돌려받을 수 있나요?"
    assert examples[0].relevant_document_ids == ["criminal::judgment::64516"]
    assert report.kept == 1 and report.total == 1


def test_gold_answer_never_reaches_the_dataset(tmp_path):
    # The whole point of keeping qa.jsonl out of the index. Guard it here too.
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row()])
    examples, _ = build_retrieval_examples(path, {"criminal::judgment::64516"})
    serialized = examples[0].model_dump_json()
    assert "임차권등기명령" not in serialized
    assert "answer" not in serialized


def test_unmatched_questions_are_dropped_and_counted(tmp_path):
    # A QA row whose document never matched a fulltext source can never be retrieved;
    # keeping it would depress Recall@K for a data-matching reason, not a retrieval one.
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row(), qa_row(document_id="criminal::judgment::99999", question="다른 질문인가요?")])
    examples, report = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert len(examples) == 1
    assert report.kept == 1 and report.dropped_unmatched == 1
    assert report.unmatched_sample == ["criminal::judgment::99999"]


def test_rows_missing_question_or_document_id_are_dropped(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row(question="   "), qa_row(document_id=""), qa_row()])
    examples, report = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert len(examples) == 1 and report.dropped_invalid == 2


def test_ids_are_deterministic_and_unique_per_question(tmp_path):
    path = tmp_path / "qa.jsonl"
    rows = [qa_row(), qa_row(question="두 번째 질문입니다.")]
    write_qa(path, rows)
    first, _ = build_retrieval_examples(path, {"criminal::judgment::64516"})
    write_qa(path, list(reversed(rows)))
    second, _ = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert {e.id for e in first} == {e.id for e in second}
    assert len({e.id for e in first}) == 2


def test_duplicate_questions_for_one_document_collapse(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row(), qa_row()])
    examples, report = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert len(examples) == 1 and report.dropped_duplicate == 1


def test_filters_are_optional(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row()])
    plain, _ = build_retrieval_examples(path, {"criminal::judgment::64516"})
    assert plain[0].filters is None
    filtered, _ = build_retrieval_examples(path, {"criminal::judgment::64516"}, with_filters=True)
    assert filtered[0].filters.legal_domain == "criminal"
    assert filtered[0].filters.document_types == ["judgment"]


def test_sampling_is_deterministic_and_bounded(tmp_path):
    path = tmp_path / "qa.jsonl"
    known = {f"d{i}" for i in range(50)}
    write_qa(path, [qa_row(document_id=f"d{i}", question=f"질문 {i}입니까?") for i in range(50)])
    a, report = build_retrieval_examples(path, known, max_examples=10, seed=7)
    b, _ = build_retrieval_examples(path, known, max_examples=10, seed=7)
    c, _ = build_retrieval_examples(path, known, max_examples=10, seed=8)
    assert len(a) == 10 and report.kept == 10 and report.total == 50
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.id for e in a] != [e.id for e in c]


def test_empty_result_is_an_error(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_qa(path, [qa_row(document_id="criminal::judgment::99999")])
    with pytest.raises(ValueError, match="no usable"):
        build_retrieval_examples(path, {"criminal::judgment::64516"})


def test_malformed_line_reports_file_and_line(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text('{"document_id": "d1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="qa.jsonl:2"):
        build_retrieval_examples(path, {"d1"})


def test_corpus_document_ids_reads_ids_without_full_validation(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "documents.jsonl").write_text(
        '{"id": "d1", "unexpected_field": 1}\n\n{"id": "d2"}\n', encoding="utf-8")
    assert corpus_document_ids(corpus) == {"d1", "d2"}


def test_corpus_document_ids_rejects_a_row_without_id(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "documents.jsonl").write_text('{"id": "d1"}\n{"no": "id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="documents.jsonl:2"):
        corpus_document_ids(corpus)
