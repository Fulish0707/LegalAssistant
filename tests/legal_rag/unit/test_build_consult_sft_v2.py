from scripts.build_consult_sft_v2 import (Draft, anchor_quote, assemble_target, check_draft, framed_rule, has_party_label,
                                          judged_ok, kept_items)
from scripts.build_consult_sft import evidence_bundle, norm

CONTEXT = ("임차인이 지출한 필요비는 임대인에게 상환을 청구할 수 있다고 봄이 상당하다. 원고는 수선업체에 800,000원을 지급하고 누수를 수선하였다."
           "\n\n[발췌 구분]\n\n"
           "다만 임대인이 수선 요청을 받고 상당한 기간 내에 수선한 경우에는 그러하지 아니하다.")
ROW = {"id": "r1", "document_id": "서울중앙지방법원-2026가단1", "title": "수선비상환", "context": CONTEXT,
       "question": "필요비 상환 청구가 가능한가요?", "answer": "임차인은 필요비 상환을 청구할 수 있습니다."}


def draft(**over):
    base = dict(
        question="세 들어 사는 집에 물이 새서 제가 수리업체를 불러 50만 원을 냈어요. 집주인에게 돌려받을 수 있나요?",
        user_facts=["누수 수리비 50만 원을 직접 냈다"], missing_fact="집주인에게 먼저 수선을 요청했는지",
        legal_issue="임차인이 지출한 수선비를 임대인에게 청구할 수 있는지",
        rule="비슷한 사건에서 법원은 임차인이 지출한 필요비는 임대인에게 상환을 청구할 수 있다고 판단했습니다.",
        application="말씀하신 수리비가 집을 쓰는 데 필요한 비용이라면 집주인에게 청구할 수 있습니다.",
        summary="수리비가 필요비에 해당한다면 집주인에게 돌려받을 수 있습니다.",
        preparations=[{"material": "수리비 영수증", "proves": "수리비 50만 원을 지출한 사실", "quote": "원고는 수선업체에 800,000원을 지급하고 누수를 수선하였다."}],
        cautions=[{"text": "집주인이 요청을 받고 제때 수선했다면 청구가 어려울 수 있습니다", "quote": "다만 임대인이 수선 요청을 받고 상당한 기간 내에 수선한 경우에는 그러하지 아니하다."}],
        follow_up_questions=["수리 전에 집주인에게 수선을 요청하셨나요?"],
        rule_quote="임차인이 지출한 필요비는 임대인에게 상환을 청구할 수 있다고 봄이 상당합니다.")
    base.update(over)
    return Draft.model_validate(base)


def test_anchor_quote_accepts_tail_changes_and_rejects_paraphrase():
    hay = norm(CONTEXT)
    assert anchor_quote("임차인이 지출한 필요비는 임대인에게 상환을 청구할 수 있다고 봄이 상당합니다.", hay) == \
        "임차인이 지출한 필요비는 임대인에게 상환을 청구할 수 있다고 봄이 상당"
    assert anchor_quote("임차인이 쓴 비용은 집주인이 돌려줘야 한다고 판단하였다.", hay) is None
    assert anchor_quote("원고는 수선을 주장하였다", "원고는 수선을 주장하였다") is None


def test_clean_draft_passes_and_assembles_a_verified_target():
    from legal_rag.verification import Verifier

    d = draft()
    assert check_draft(d, ROW) == []
    preps, cauts, dropped = kept_items(d, ROW)
    assert len(preps) == 1 and len(cauts) == 1 and dropped == []
    target = assemble_target(d, ROW, preps, cauts)
    assert target.additional_information_needed == ["수리 전에 집주인에게 수선을 요청하셨나요?"]
    assert target.possible_actions == ["수리비 영수증: 수리비 50만 원을 지출한 사실"]
    assert "집주인에게 먼저 수선을 요청했는지에 따라 결론이 달라질 수 있습니다." in target.uncertainty
    assert [c.source_id for c in target.citations] == ["S1", "S2"]
    assert Verifier().verify(target, evidence_bundle(ROW)).ok


def test_party_labels_and_definitive_summary_are_rejected():
    reasons = check_draft(draft(question="원고인 제가 수리비 50만 원을 냈는데 돌려받을 수 있나요? 집주인이 안 줘요.",
                                application="이 사건에서 집주인은 수리비를 줘야 합니다.", summary="집주인은 수리비를 지급해야 합니다."), ROW)
    assert {"party_label_in_question", "party_label_in_application", "summary_not_conditional"} <= set(reasons)


def test_foreign_authority_is_rejected():
    assert "foreign_authority" in check_draft(draft(application="민법 제750조에 따라 청구할 수 있습니다."), ROW)


def test_rows_about_abolished_offences_are_dropped():
    from scripts.build_consult_sft_v2 import current_law_rows

    rows = [{"title": "간통", "question": "고소할 수 있나요?", "answer": "..."}, ROW]
    assert current_law_rows(rows) == [ROW]


def test_load_review_and_strict_ok(tmp_path):
    from scripts.build_consult_sft_v2 import load_review, strict_ok

    path = tmp_path / "review.jsonl"
    path.write_text('{"id": "a", "keep": true, "reason": "ok"}\n{"id": "b", "keep": false, "reason": "x"}\n')
    review = load_review(path)
    assert review is not None and review["a"]["keep"] and not review["b"]["keep"]
    assert load_review(None) is None
    full = dict.fromkeys(("question_coherent", "no_fabricated_user_fact", "rule_matches_source",
                          "application_follows_rule", "conclusion_conditional"), True)
    assert strict_ok(full) and not strict_ok({**full, "rule_matches_source": False})


def test_template_target_uses_reference_answer_and_stays_conditional():
    from legal_rag.verification import Verifier
    from scripts.build_consult_sft_v2 import particle, template_problems, template_target

    d = draft()
    assert template_problems(d, ROW) == []
    preps, cauts, _ = kept_items(d, ROW)
    t = template_target(d, ROW, preps, cauts)
    assert t.analysis.startswith("비슷한 사건에서 법원은 다음과 같이 판단했습니다. 임차인은 필요비 상환을 청구할 수 있습니다.")
    assert "말씀하신 상황(누수 수리비 50만 원을 직접 냈다)" in t.analysis
    assert t.summary.endswith("집주인에게 먼저 수선을 요청했는지를 확인해야 합니다.")
    assert t.additional_information_needed and t.citations
    assert Verifier().verify(t, evidence_bundle(ROW)).ok
    assert particle("여부", "을", "를") == "를" and particle("사실", "을", "를") == "을"


def test_template_v3_removes_cross_field_repetition():
    from legal_rag.verification import Verifier
    from scripts.build_consult_sft_v2 import template_target, template_target_v3

    d = draft(cautions=[{"text": "집주인이 제때 수선했다면 청구가 어렵습니다", "quote": "다만 임대인이 수선 요청을 받고 상당한 기간 내에 수선한 경우에는 그러하지 아니하다."},
                        {"text": "집주인이 제때 수선했다면 청구가 어렵습니다", "quote": "다만 임대인이 수선 요청을 받고 상당한 기간 내에 수선한 경우에는 그러하지 아니하다."}],
              follow_up_questions=["수리 전에 집주인에게 수선을 요청하셨나요?", "수리 전에 집주인에게 수선을 요청하셨나요?"])
    preps, cauts, _ = kept_items(d, ROW)
    v2, v3 = template_target(d, ROW, preps, cauts), template_target_v3(d, ROW, preps, cauts)
    assert v2.analysis.count("집주인이 제때 수선했다면") > 1 and v3.analysis.count("집주인이 제때 수선했다면") == 1
    assert "집주인이 제때 수선했다면" not in v3.uncertainty
    assert len(v3.additional_information_needed) == 1
    assert v3.summary == v2.summary and len(v3.analysis) < len(v2.analysis)
    assert v3.analysis.split("\n\n주의할 점:")[0] == v2.analysis.split("\n\n주의할 점:")[0]
    assert Verifier().verify(v3, evidence_bundle(ROW)).ok


def test_template_problems_flag_case_specific_reference_and_unstated_facts():
    from scripts.build_consult_sft_v2 import template_problems

    assert "reference_has_party_labels" in template_problems(draft(), {**ROW, "answer": "피고는 원고에게 지급하라."})
    assert "user_fact_not_in_question" in template_problems(draft(user_facts=["집주인과 계약서를 두 번 갱신했다"]), ROW)


def test_framed_rule_and_judge():
    assert framed_rule("임차인은 필요비를 청구할 수 있습니다.") == "관련 판결에 따르면, 임차인은 필요비를 청구할 수 있습니다."
    assert framed_rule("법원은 청구를 인정했습니다.") == "법원은 청구를 인정했습니다."
    assert has_party_label("피고인이 말했다") and not has_party_label("집주인이 말했다")
    assert judged_ok({"fabricated_user_fact": False, "rule_supported": True, "conclusion_conditional": True})
    assert not judged_ok({"fabricated_user_fact": True, "rule_supported": True, "conclusion_conditional": True})
    assert not judged_ok(None)
