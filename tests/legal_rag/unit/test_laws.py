from legal_rag.retrieval.laws import articles, cited_articles, compact_names
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType


def part(order, article, text, paragraph=None):
    return Chunk(id=f"law#art:{article}" + (f"/para:{paragraph}" if paragraph else ""), document_id="law",
                 document_type=DocumentType.statute, path=str(order), order=order, text=text, title="주택임대차보호법",
                 metadata=ChunkMetadata(dataset="f", corpus_version="v", law_name="주택임대차보호법", article=article,
                                        paragraph=paragraph))


LAWS = compact_names(["주택임대차보호법", "상가건물 임대차보호법", "민법", "난민법", "민사집행법"])


def cited(text):
    return [(ref.law_name, ref.article, ref.paragraph) for ref in cited_articles(text, LAWS)]


def test_a_citation_takes_the_longest_known_law_name_before_it():
    assert cited("임차인은 주택임대차보호법 제3조의3 제1항에 따라 신청할 수 있고, 난민법 제2조는 무관하다.") == [
        ("주택임대차보호법", "제3조의3", "제1항"), ("난민법", "제2조", None)]
    assert cited("상가건물임대차보호법 제10조 및 「상가건물 임대차보호법」 제10조의4") == [
        ("상가건물 임대차보호법", "제10조", None), ("상가건물 임대차보호법", "제10조의4", None)]


def test_a_citation_without_a_name_belongs_to_the_law_named_before_it():
    text = "주택임대차보호법 제3조 제1항의 대항력은 같은 법 제3조의2 제2항, 제4조와 구별된다. 민법 제618조, 법 제623조"

    assert cited(text) == [("주택임대차보호법", "제3조", "제1항"), ("주택임대차보호법", "제3조의2", "제2항"),
                           ("주택임대차보호법", "제4조", None), ("민법", "제618조", None), ("민법", "제623조", None)]


def test_an_unknown_law_and_the_bare_citations_after_it_are_dropped():
    text = "민법 제618조. 구 임대주택법 제21조, 제22조 제1항은 폐지되었다. 호법 제3조 제4항. 민사집행법 제283조"

    assert cited(text) == [("민법", "제618조", None), ("민사집행법", "제283조", None)]


def test_a_bare_citation_before_any_law_is_dropped_and_repeats_are_listed_once():
    assert cited("제5조에 따라, 민법 제618조와 민법 제618조") == [("민법", "제618조", None)]


def test_an_article_in_one_chunk_keeps_its_heading_apart_from_its_text():
    listed = articles([part(0, "제3조", "주택임대차보호법 제3조(대항력 등)\n제①항 임대차는 효력이 생긴다.\n제②항 준용한다.")])

    assert listed == [{"article": "제3조", "title": "대항력 등", "text": "제①항 임대차는 효력이 생긴다.\n제②항 준용한다.",
                       "chunk_id": "law#art:제3조"}]


def test_an_article_stored_by_paragraph_is_joined_under_its_paragraph_labels():
    chunks = [
        part(0, "제3조의3", "주택임대차보호법 제3조의3(임차권등기명령) 제①항\n임차인은 신청할 수 있다.", "제①항"),
        part(1, "제3조의3", "주택임대차보호법 제3조의3(임차권등기명령) 제②항\n신청서에는 다음을 적어야 한다.\n    1. 신청의 취지", "제②항"),
        part(2, "제4조", "주택임대차보호법 제4조(임대차기간 등)\n기간은 2년으로 본다."),
    ]

    listed = articles(chunks)

    assert [(a["article"], a["title"], a["chunk_id"]) for a in listed] == [
        ("제3조의3", "임차권등기명령", "law#art:제3조의3/para:제①항"), ("제4조", "임대차기간 등", "law#art:제4조")]
    assert listed[0]["text"] == "제①항 임차인은 신청할 수 있다.\n제②항 신청서에는 다음을 적어야 한다.\n    1. 신청의 취지"


def test_an_article_without_a_heading_has_an_empty_title():
    assert articles([part(0, "제1조", "어떤 규칙 제1조\n이 규칙은 목적을 정한다.")])[0]["title"] == ""
