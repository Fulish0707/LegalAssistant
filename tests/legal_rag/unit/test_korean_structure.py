"""The text-structure parser for imported Korean legal documents."""

from __future__ import annotations

import re

import pytest

from legal_rag.chunking.korean_structure import (
    ParsedStructure,
    Section,
    parse_structure,
    split_korean_sentences,
)

STATUTE = """제1조(목적)
이 법은 상표를 보호함을 목적으로 한다.
제2조(정의)
① 이 법에서 사용하는 용어의 뜻은 다음과 같다.
1. "상표"란 자기의 상품을 식별하기 위하여 사용하는 표장을 말한다.
2. "표장"이란 기호, 문자, 도형 등을 말한다.
② 단체표장에 관하여는 이 법에서 특별히 규정한 것을 제외한다.
제3조의2(특례)
특례를 정한다.
"""

DECISION = """이 사건 심판청구를 기각한다.
원결정을 취소한다.
1. 기초사실
가. 절차의 경위
(1) 특허출원 : 2008. 12. 31.
(2) 거절결정 : 2011. 03. 30.
나. 이 사건 출원발명의 요지
발명의 요지는 다음과 같다.
2. 청구인 주장의 요지
청구인은 이렇게 주장한다.
3. 판단
가. 진보성
진보성이 인정되지 아니한다.
"""

INTERPRETATION = """질의요지 「저작권법」 제7조제1호에 초안이 포함되는지?
회답 포함되지 아니합니다.
이유 「저작권법」 제2조제1호에서는 저작물을 정의하고 있습니다.
그러므로 포함되지 아니합니다.
"""


def _leaf_texts(parsed: ParsedStructure) -> list[str]:
    out: list[str] = []

    def walk(section: Section) -> None:
        if section.body:
            out.append(section.body)
        for child in section.children:
            walk(child)

    for section in parsed.sections:
        walk(section)
    return out


SYNTHETIC = {"주문·청구취지", "전문"}


def _words(text: str) -> list[str]:
    """Letters and digits only: headings may lose surrounding brackets or colons."""
    return sorted(re.findall(r"[0-9A-Za-z\uac00-\ud7a3]", text))


def _coverage(parsed: ParsedStructure, text: str) -> None:
    """Every letter of the source survives, once, in a heading or exactly one body."""
    headings = "".join(s.heading for s in _all_sections(parsed) if s.label not in SYNTHETIC)
    assert _words(headings + "".join(_leaf_texts(parsed))) == _words(text)


def _all_sections(parsed: ParsedStructure) -> list[Section]:
    out: list[Section] = []

    def walk(section: Section) -> None:
        out.append(section)
        for child in section.children:
            walk(child)

    for section in parsed.sections:
        walk(section)
    return out


class TestSentenceSplitting:
    def test_keeps_korean_dotted_dates_whole(self):
        assert split_korean_sentences("특허출원은 2008. 12. 31.에 있었다. 그 후 보정되었다.") == [
            "특허출원은 2008. 12. 31.에 있었다.",
            "그 후 보정되었다.",
        ]

    def test_does_not_split_article_references(self):
        assert split_korean_sentences("제3조 제1항에 따른다.") == ["제3조 제1항에 따른다."]

    def test_splits_ordinary_sentences(self):
        assert len(split_korean_sentences("가나다 한다. 라마바 한다. 사아자 한다.")) == 3

    def test_empty_text(self):
        assert split_korean_sentences("   ") == []


class TestStatuteParsing:
    def test_articles_become_top_level_sections(self):
        parsed = parse_structure(STATUTE, "statute")
        assert parsed is not None and parsed.kind == "statute"
        assert [s.label for s in parsed.sections] == ["제1조", "제2조", "제3조의2"]
        assert parsed.sections[0].heading == "제1조(목적)"
        assert parsed.sections[2].heading == "제3조의2(특례)"

    def test_paragraphs_become_children(self):
        parsed = parse_structure(STATUTE, "statute")
        article = parsed.sections[1]
        assert [c.label for c in article.children] == ["①", "②"]
        assert "상표" in article.children[0].body

    def test_article_without_paragraphs_keeps_body(self):
        parsed = parse_structure(STATUTE, "statute")
        assert parsed.sections[0].children == []
        assert parsed.sections[0].body.startswith("이 법은")

    def test_covers_every_character(self):
        _coverage(parse_structure(STATUTE, "statute"), STATUTE)

    def test_unparseable_statute_returns_none(self):
        assert parse_structure("조문 없는 평범한 문단입니다.\n두 번째 문단입니다.", "statute") is None


class TestNumberedParsing:
    def test_preamble_becomes_its_own_section(self):
        parsed = parse_structure(DECISION, "patent_trial_decision")
        assert parsed is not None and parsed.kind == "numbered"
        assert parsed.sections[0].heading == "주문·청구취지"
        assert "기각한다" in parsed.sections[0].body

    def test_three_heading_levels_nest(self):
        parsed = parse_structure(DECISION, "patent_trial_decision")
        facts = parsed.sections[1]
        assert facts.heading == "1. 기초사실"
        assert [c.heading for c in facts.children] == ["가. 절차의 경위", "나. 이 사건 출원발명의 요지"]
        assert [c.label for c in facts.children[0].children] == ["(1)", "(2)"]

    def test_named_headings_are_recognised(self):
        text = "피고인을 벌금에 처한다.\n1. 항소이유\n주장이 있다.\n범죄사실\n피고인은 침해하였다.\n증거의 요지\n제1심과 같다.\n"
        parsed = parse_structure(text, "judgment")
        assert [s.heading for s in parsed.sections] == [
            "주문·청구취지",
            "1. 항소이유",
            "범죄사실",
            "증거의 요지",
        ]

    def test_covers_every_character(self):
        _coverage(parse_structure(DECISION, "patent_trial_decision"), DECISION)

    def test_needs_at_least_two_headings(self):
        assert parse_structure("1. 기초사실\n사실은 이러하다.\n", "judgment") is None

    def test_unstructured_text_returns_none(self):
        assert parse_structure("평범한 문단입니다.\n두 번째 문단입니다.\n", "judgment") is None


class TestInterpretationParsing:
    def test_markers_become_sections(self):
        parsed = parse_structure(INTERPRETATION, "legal_interpretation")
        assert parsed is not None and parsed.kind == "interpretation"
        assert [s.label for s in parsed.sections] == ["질의요지", "회답", "이유"]
        assert parsed.sections[1].body == "포함되지 아니합니다."

    def test_trailing_lines_stay_with_their_marker(self):
        parsed = parse_structure(INTERPRETATION, "legal_interpretation")
        assert "그러므로" in parsed.sections[2].body

    def test_covers_every_character(self):
        _coverage(parse_structure(INTERPRETATION, "legal_interpretation"), INTERPRETATION)

    def test_missing_markers_return_none(self):
        assert parse_structure("그냥 해설입니다.\n두 번째 줄입니다.\n", "legal_interpretation") is None


class TestParseGuards:
    @pytest.mark.parametrize("text", ["", "   \n\n  "])
    def test_empty_text_is_unparseable(self, text):
        assert parse_structure(text, "judgment") is None

    def test_unknown_document_type_uses_the_numbered_parser(self):
        assert parse_structure(DECISION, "decision") is not None
