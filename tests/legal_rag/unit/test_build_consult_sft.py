import json

from scripts.build_consult_sft import (Item, Preparation, answer_laws_grounded, assemble_answer, build_citations,
                                       evidence_bundle, first_sentence, is_court_judgment, is_material, issue_from_question,
                                       norm, passages,
                                       production_messages, validate_items)

CONTEXT = ("제623조(임대인의 의무) 임대인은 목적물을 임차인에게 인도하고 계약존속중 그 사용, 수익에 필요한 상태를 유지하게 할 의무를 부담한다."
           "\n\n[발췌 구분]\n\n"
           "원고는 수선업체에 800,000원을 지급하고 누수를 수선하였다. 피고는 요청을 받고도 수선하지 않았다.")


def test_validate_keeps_literal_quotes_and_drops_inventions():
    items = [
        Item(text="수선비 영수증", quote="원고는 수선업체에 800,000원을 지급하고"),
        Item(text="반환 의사 확인서", quote="피고는 반환 의사를 확인하였다"),           # not in passages
        Item(text="민법 제750조 손해배상 청구서", quote="피고는 요청을 받고도 수선하지 않았다"),  # foreign statute
        Item(text="수선비 영수증", quote="원고는 수선업체에 800,000원을 지급하고"),      # duplicate
    ]
    kept, dropped = validate_items(items, CONTEXT)
    assert [k.text for k in kept] == ["수선비 영수증"]
    assert [d["reason"].split(":")[0] for d in dropped] == ["quote_not_in_passages", "foreign_authority", "duplicate"]


def test_validate_tolerates_whitespace_and_quote_marks():
    kept, _ = validate_items([Item(text="수선 요청 기록", quote="“피고는 요청을   받고도 수선하지 않았다”")], CONTEXT)
    assert len(kept) == 1 and kept[0].quote == "피고는 요청을 받고도 수선하지 않았다"


def test_assemble_keeps_reference_and_places_items():
    answer = assemble_answer("필요비는 상환을 청구할 수 있습니다. 다만 요건이 있습니다.",
                             [Item(text="수선비 영수증", quote="원고는 수선업체에 800,000원을 지급하고")],
                             [Item(text="임대인이 이미 수선했으면 청구할 수 없음", quote="피고는 요청을 받고도 수선하지 않았다")])
    assert answer.analysis.startswith("필요비는 상환을 청구할 수 있습니다.")
    assert "주의할 점: 임대인이 이미 수선했으면 청구할 수 없음." in answer.analysis
    assert answer.summary == "필요비는 상환을 청구할 수 있습니다."
    assert answer.possible_actions == ["수선비 영수증"]
    assert answer.citations == []


def test_assemble_without_items_is_reference_only():
    answer = assemble_answer("결론입니다.", [], [])
    assert answer.analysis == "결론입니다." and answer.possible_actions == [] and answer.uncertainty == ""


def test_preparation_renders_material_and_fact():
    item = Preparation(material="수선비 영수증", proves="수선비를 지출한 사실", quote="원고는 수선업체에 800,000원을 지급하고").item()
    assert item.text == "수선비 영수증: 수선비를 지출한 사실"


def test_answer_law_filter_rejects_foreign_law_names():
    context = "방송광고판매대행 등에 관한 법률 제5조 ② 한국방송공사는 위탁하는 방송광고에 한정하여 방송광고를 할 수 있다."
    assert answer_laws_grounded("방송광고판매대행 등에 관한 법률 제5조에 따르면 가능합니다.", context, "방송광고판매대행 등에 관한 법률")
    assert not answer_laws_grounded("민원 처리에 관한 법률 제5조 제2항에 의하면 가능합니다.", context, "방송광고판매대행 등에 관한 법률")


def test_answer_law_filter_accepts_spaced_names_and_rejects_foreign_articles():
    context = "농업기계화 촉진법 제12조의2(안전교육) ① 농림축산식품부장관은 안전교육계획을 매년 수립하고 시행하여야 한다."
    assert answer_laws_grounded("농업기계화 촉진법 제12조의2 제1항에 따르면 매년 수립합니다.", context, "농업기계화 촉진법")
    assert not answer_laws_grounded("농업기계화 촉진법 제13조에 따르면 매년 수립합니다.", context, "농업기계화 촉진법")


def test_is_material_rejects_statutes_and_the_judgment_itself():
    assert is_material("수선비 영수증") and is_material("감정 결과") and is_material("문자 기록")
    assert is_material("기사에 달린 댓글") and is_material("채권자목록") and is_material("피고인의 진술")
    for name in ("민법 제827조 제1항", "판결문", "법률자료 발췌문", "청구인의 주장", "항소이유에 관한 판단", "상표법",
                 "대법원 2012.11.15. 선고 2010도6343 판", "민원사무처리기준표 제9항", "채권자와 채무자의 관계"):
        assert not is_material(name), name


def test_only_court_judgments_are_sampled():
    assert is_court_judgment("criminal::judgment::77482")
    assert is_court_judgment("서울중앙지방법원-2017가단27240")
    assert not is_court_judgment("administrative::statute::000454")
    assert not is_court_judgment("administrative::constitutional_decision::168593")
    assert not is_court_judgment("2013100001331")


def test_issue_from_question_drops_only_the_interrogative():
    assert issue_from_question("채권자대위권을 행사하기 위해 요구되는 조건은 무엇인가요?") == "채권자대위권을 행사하기 위해 요구되는 조건"
    assert issue_from_question("공동불법행위에서 행위자 상호간의 공모가 필요한가요?") == "공동불법행위에서 행위자 상호간의 공모가 필요한지"
    assert issue_from_question("공포되지 않았을 경우에도 그 효력이 유지되나요?") == "공포되지 않았을 경우에도 그 효력이 유지되는지"
    assert issue_from_question("손해의 범위는 어떻게 결정되나요?") == "손해의 범위는 어떻게 결정되는지"


def test_citations_point_at_the_passage_holding_each_quote():
    row = {"document_id": "서울중앙지방법원-2017가단27240", "title": "수선비상환", "context": CONTEXT, "answer": "수선비를 청구할 수 있다."}
    cites = build_citations(row, ["피고는 요청을 받고도 수선하지 않았다"])
    assert [(c.source_id, c.reference) for c in cites] == [("S2", "서울중앙지방법원 2017가단27240")]
    fallback = build_citations({**row, "document_id": "civil::judgment::9"}, [])
    assert len(fallback) == 1 and fallback[0].reference == "수선비상환 판결" and (fallback[0].quote or "") in norm(CONTEXT)


def test_assembled_target_passes_the_production_verifier():
    from legal_rag.verification import Verifier

    row = {"id": "r1", "document_id": "서울중앙지방법원-2017가단27240", "title": "수선비상환", "context": CONTEXT,
           "answer": "민법 제623조에 따라 임대인은 수선 의무가 있습니다.", "question": "수선비를 받을 수 있나요?"}
    items = [Item(text="수선비 영수증: 지출 사실", quote="원고는 수선업체에 800,000원을 지급하고")]
    target = assemble_answer(row["answer"], items, [], question=row["question"],
                             citations=build_citations(row, [i.quote for i in items]))
    assert Verifier().verify(target, evidence_bundle(row)).ok
    assert target.legal_issues == ["수선비를 받을 수 있는지"]


def test_first_sentence_and_passages():
    assert first_sentence("첫 문장입니다. 둘째 문장입니다.") == "첫 문장입니다."
    assert len(passages(CONTEXT)) == 2


def test_production_messages_match_serving_prompt():
    from legal_rag.llm.prompts import load_prompt

    row = {"id": "r1", "document_id": "civil::judgment::1", "title": "수선비상환", "context": CONTEXT}
    target = assemble_answer("결론입니다.", [], [])
    messages = production_messages("돌려받을 수 있나요?", row, target)
    assert messages[0] == {"role": "system", "content": load_prompt("answer_system")}
    assert "[LEGAL SOURCE S1]" in messages[1]["content"] and "[LEGAL SOURCE S2]" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "질문: 돌려받을 수 있나요?"}
    assert json.loads(target.model_dump_json())["analysis"] == "결론입니다."
