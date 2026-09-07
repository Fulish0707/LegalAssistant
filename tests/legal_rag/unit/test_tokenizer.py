import pytest

from legal_rag.retrieval.tokenizer import RegexTokenizer, build_tokenizer


def test_preserves_legal_identifiers():
    toks = RegexTokenizer().tokenize("민법 제750조와 주택임대차보호법 제 3 조의2, 2018다12345 판결")
    assert "제750조" in toks and "제3조의2" in toks and "2018다12345" in toks
    assert "민법" in toks


def test_emits_hangul_bigrams_and_whole_words():
    toks = RegexTokenizer().tokenize("보증금반환")
    assert toks == ["보증금반환", "보증", "증금", "금반", "반환"]


def test_lowercases_ascii_and_drops_punctuation():
    assert RegexTokenizer().tokenize("RAG, Qdrant!") == ["rag", "qdrant"]


def test_build_tokenizer_unknown():
    with pytest.raises(ValueError):
        build_tokenizer("nope")
    assert build_tokenizer("regex").name == "regex"
