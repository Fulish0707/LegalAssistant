"""Quote-position roles for windowed case text. The rule is rule_v1 from the labelled
evaluation (~/.hermes/profiles/kei/work/role-eval): headings carried across the document
plus sentence cues. It is a filter against party claims, not a finding classifier."""

from legal_rag.chunking.quote_roles import LEGAL_SUPPORT_ROLES, quote_role, sentence_at


def role_of(text, sentence):
    pos = text.index(sentence)
    return quote_role(text, pos, sentence_at(text, pos))


def test_claim_restated_in_a_judgment_section_is_a_claim():
    text = "3. 판단\n가. 해지 여부\n피고는 원고가 계약을 위반하였으므로 즉시 해지할 수 있다고 주장한다. 그러나 피고의 주장은 타당하지 않다."
    assert role_of(text, "피고는 원고가") == "party_claim"


def test_court_rejecting_a_claim_is_a_finding():
    text = "3. 판단\n원고는 계약이 무효라고 주장하나, 이를 인정할 증거가 없으므로 원고의 주장은 이유 없다."
    assert role_of(text, "원고는 계약이") == "finding"


def test_speech_inside_an_open_quote_is_not_the_court():
    text = "2. 판단\n피해자는 ‘피고인을 그날 처음 보았다. 피고인이 차용증을 보여주어 사인을 했다."
    assert role_of(text, "피고인이 차용증을") == "quoted"


def test_reason_heading_glued_to_the_order_line_ends_the_order_section():
    text = "[주 문]\n이 사건 심판청구를 각하한다.[이 유]\n청구인은 같은 취지의 재심을 반복하여 청구하고 있다."
    assert role_of(text, "청구인은 같은") != "order"


def test_bare_spaced_titles_are_headings():
    text = "주 문\n피고인을 징역 5월에 처한다.\n이 유\n범 죄 사 실\n피고인은 피해자를 기망하였다.\n양형의 이유\n피고인이 범행을 인정하고 있다."
    assert role_of(text, "피고인은 피해자를") == "facts"
    assert role_of(text, "피고인이 범행을") == "finding"


def test_appeal_grounds_summary_is_a_claim_but_ruling_on_them_is_a_finding():
    claim = "1. 항소이유의 요지\n원심의 형이 너무 무거워 부당하다."
    ruling = "2. 상고이유에 대하여\n원심의 판단은 정당하다."
    assert role_of(claim, "원심의 형이") == "party_claim"
    assert role_of(ruling, "원심의 판단은") == "finding"


def test_dates_and_masked_figures_are_not_headings():
    text = "3. 판단\n가.1975.7.9. 상피고인으로부터 공사금을 양수하였다.\n가.(날짜) 상피고인으로부터 대금을 받았다.\n따라서 피고인의 행위는 횡령에 해당한다."
    assert role_of(text, "따라서 피고인의") == "finding"


def test_text_without_headings_is_unknown_and_unknown_still_counts_as_support():
    text = "청구인의 남편은 유죄판결을 받았고 청구인은 헌법소원심판을 청구하였다."
    assert role_of(text, "청구인의 남편은") == "unknown"
    assert "unknown" in LEGAL_SUPPORT_ROLES
    assert not {"party_claim", "quoted", "facts", "order", "evidence", "mixed"} & LEGAL_SUPPORT_ROLES


def test_sentence_at_handles_sentences_glued_without_a_space():
    text = "벌금을 납부하여야 한다.제117조는 범칙자를 정의한다."
    assert sentence_at(text, text.index("제117조")) == "제117조는 범칙자를 정의한다."
