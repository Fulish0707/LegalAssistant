from legal_rag.evaluation.answers.reference_overlap import bigram_recall, fact_recall, paired_difference


def test_bigram_recall_ignores_spacing_and_counts_repeats_once_each():
    assert bigram_recall("구제신청은 3개월 이내", "구제 신청은 3개월 이내") == 1.0
    assert bigram_recall("전혀 다른 문장", "구제신청은 3개월 이내") == 0.0
    assert 0.0 < bigram_recall("구제신청은 가능합니다", "구제신청은 3개월 이내") < 1.0
    assert bigram_recall("아무 답변", "") == 0.0


def test_fact_recall_looks_for_the_references_articles_and_quantities():
    reference = "근로기준법 제28조 제2항에 따라 3개월 이내에 신청해야 합니다."
    assert fact_recall("제28조에 따르면 3개월 안에 신청합니다.", reference) == 1.0
    assert fact_recall("제27조에 따르면 3개월 안에 신청합니다.", reference) == 0.5
    assert fact_recall("답변", "조문도 수치도 없는 참고 답변") is None


def test_paired_difference_reports_mean_and_a_deterministic_interval():
    base, new = [0.5, 0.6, 0.4, 0.7], [0.6, 0.7, 0.5, 0.8]
    mean, (low, high) = paired_difference(base, new)
    assert round(mean, 3) == 0.1 and round(low, 3) == 0.1 and round(high, 3) == 0.1
    assert paired_difference(base, new) == paired_difference(base, new)
