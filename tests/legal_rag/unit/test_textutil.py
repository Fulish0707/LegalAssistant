from datetime import date

from legal_rag.textutil.dates import parse_korean_date
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs, normalize_article
from legal_rag.textutil.sentences import split_sentences


def test_statute_refs_with_law_name_and_sub_article():
    text = "주택임대차보호법 제3조의2 제2항과 민법 제750조에 따라"
    refs = extract_statute_refs(text)
    assert [r.render() for r in refs] == ["주택임대차보호법 제3조의2", "민법 제750조"]
    assert refs[0].paragraph == "제2항"


def test_bare_article_without_law_name():
    refs = extract_statute_refs("같은 법 제10조에 의하면")
    assert refs[0].law_name is None and refs[0].article == "제10조"


def test_case_refs_ignore_dates():
    text = "대법원 2019. 5. 30. 선고 2018다12345 판결, 2020도1234, 2021년12월"
    assert extract_case_refs(text) == ["2018다12345", "2020도1234"]


def test_normalize_article_spacing():
    assert normalize_article("제 3 조의 2") == "제3조의2"


def test_parse_dates():
    assert parse_korean_date("2019.01.02") == date(2019, 1, 2)
    assert parse_korean_date("20190102") == date(2019, 1, 2)
    assert parse_korean_date("2019. 1. 2.") == date(2019, 1, 2)
    assert parse_korean_date("2019-01-02") == date(2019, 1, 2)
    assert parse_korean_date("") is None
    assert parse_korean_date("2019.13.40") is None


def test_split_sentences_korean():
    text = "임차인은 보증금을 돌려받을 수 있다. 다만 등기명령을 먼저 신청해야 한다! 정말인가?"
    assert split_sentences(text) == [
        "임차인은 보증금을 돌려받을 수 있다.",
        "다만 등기명령을 먼저 신청해야 한다!",
        "정말인가?",
    ]


def test_case_refs_cover_constitutional_court_numbers():
    """헌법재판소 numbers use two-character types (헌마, 헌바, ...)."""
    assert extract_case_refs("사  건○  2020헌마843 가석방") == ["2020헌마843"]
    assert extract_case_refs("2018헌바434 위헌소원, 2023헌가12") == ["2018헌바434", "2023헌가12"]
