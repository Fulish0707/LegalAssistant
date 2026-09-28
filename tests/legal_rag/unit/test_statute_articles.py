import json

from legal_rag.ingestion.corpus import Corpus
from legal_rag.ingestion.statute_articles import StatuteArticleLocator
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType

LAW = (
    "제1장 총칙\n"
    "제1조(목적) 이 법은 목적을 정한다.\n"
    "제2조(정의) 이 법에서 쓰는 말의 뜻은 다음과 같다.\n1. 부단체장\n2. 제3조제2항에 따른 사람\n"
    "제3조 삭제 <2010. 1. 1.>\n"
    "제4조(권한대행) 지방자치단체의 장이 궐위된 경우 부단체장이 그 권한을 대행한다.\n"
    "제4조의2(보고) 권한대행자는 보고하여야 한다.\n"
    "부칙\n제1조(시행일) 이 법은 공포한 날부터 시행한다.\n"
)


def _windows(*starts: str) -> list[str]:
    positions = [LAW.index(start) for start in starts] + [len(LAW)]
    return [LAW[a:b] for a, b in zip(positions, positions[1:])]


def test_window_opening_mid_article_is_named_after_it():
    """지방자치법 chunk 13 opened in the body of 제124조, whose heading was in chunk 12 (2026-09-28)."""
    locator = StatuteArticleLocator(LAW)
    assert [locator.article(w) for w in _windows("제1장", "1. 부단체장", "지방자치단체의 장이", "권한대행자는")] == [
        "제1조", "제2조", "제4조", "제4조의2"]


def test_cross_reference_at_a_line_start_is_not_a_heading():
    assert StatuteArticleLocator(LAW).article(LAW[LAW.index("2. 제3조제2항"):]) == "제2조"


def test_deleted_article_counts_as_a_heading():
    assert StatuteArticleLocator(LAW).article(LAW[LAW.index("<2010"):]) == "제3조"


def test_addenda_do_not_name_the_main_text():
    assert StatuteArticleLocator(LAW).article(LAW[LAW.index("이 법은 공포한"):]) is None


def test_overlapping_windows_use_their_own_start():
    """Windows may reach back before the previous window's last heading."""
    locator = StatuteArticleLocator(LAW)
    assert locator.article(LAW[LAW.index("제2조"):]) == "제2조"
    assert locator.article(LAW[LAW.index("2. 제3조제2항"):]) == "제2조"


def _chunk(order: int, text: str, article: str | None = None, document_type=DocumentType.statute) -> Chunk:
    return Chunk(id=f"d#{order}", document_id="d", document_type=document_type, path=f"chunk:{order:04d}", order=order,
                 text=text, title="지방자치법",
                 metadata=ChunkMetadata(dataset="aihub_preprocessed", corpus_version="v1", law_name="지방자치법", article=article))


def test_corpus_fills_the_article_of_imported_statute_windows(tmp_path):
    windows = _windows("제1장", "지방자치단체의 장이")
    (tmp_path / "documents.jsonl").write_text(
        json.dumps({"id": "d", "document_type": "statute", "text": LAW}, ensure_ascii=False) + "\n", encoding="utf-8")
    chunks = [_chunk(0, windows[0]), _chunk(1, windows[1]), _chunk(2, "제9조(기존) 본문", article="제9조")]
    (tmp_path / "chunks.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    assert [c.metadata.article for c in Corpus.load(tmp_path).chunks()] == ["제1조", "제4조", "제9조"]
