"""Build the held-out answer-prompt comparison inputs (data/eval/answer_prompt_heldout).

Eight NEW cases (lease 2, civil 2, criminal 2, admin 2) with new issues, new fictional
judgments and new statutes, none reused from the development set's fact patterns.
The comparison table is fixed here, before any model run. Adoption gates are in README.md.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src .venv/bin/python scripts/build_answer_prompt_heldout.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_answer_eval import FICTIONAL_MARK, article_text  # noqa: E402

OUT = ROOT / "data/eval/answer_prompt_heldout"
CORPUS = ROOT / "data/api_laws/statute-api-full"

STATUTES = [
    ("law-646", "민법", "제646조"), ("law-629", "민법", "제629조"),
    ("law-580", "민법", "제580조"), ("law-582", "민법", "제582조"), ("law-766", "민법", "제766조"),
    ("law-c307", "형법", "제307조"), ("law-c310", "형법", "제310조"),
    ("law-c355", "형법", "제355조"), ("law-c356", "형법", "제356조"),
    ("law-ap27", "행정심판법", "제27조"), ("law-id9", "공공기관의 정보공개에 관한 법률", "제9조"),
]

JUDGMENTS = [
    ("case-HL1", "부속물매수청구", "2026가단99101",
     "원고는 피고 소유 건물을 3년간 임차하면서, 피고의 동의를 얻어 사용의 편익을 위하여 시스템 에어컨을 설치하였다. 임대차 종료 시 원고는 피고에게 에어컨의 매수를 청구하였다.",
     "피고는 에어컨이 원고의 재산이므로 매수 의무가 없다고 주장한다. 원고는 동의를 얻어 부속한 물건이므로 매수 청구권이 있다고 주장한다.",
     "건물의 임차인이 그 사용의 편익을 위하여 임대인의 동의를 얻어 건물에 부속한 물건이 있는 때에는, 임대차 종료 시 임대인에게 그 부속물의 매수를 청구할 수 있다(민법 제646조 제1항). 이 사건 에어컨은 피고의 동의를 얻어 건물의 사용 편익을 위하여 설치된 부속물에 해당하므로, 원고의 매수 청구는 이유 있다.",
     "피고는 에어컨을 현황 시가로 매수한다."),
    ("case-HL2", "무단전대해지", "2026가단99102",
     "원고는 피고 소유 주택을 임차하여 거주하던 중, 피고의 동의 없이 주택의 절반을 A에게 전대하였다. 피고는 이 사실을 알고 임대차 계약 해지를 통지하였다.",
     "원고는 일부 전대는 해지 사유가 아니라고 주장한다. 피고는 동의 없는 전대이므로 해지할 수 있다고 주장한다.",
     "임차인은 임대인의 동의 없이 임차물을 전대하지 못한다(민법 제629조 제1항). 임차인이 이를 위반한 때에는 임대인은 계약을 해지할 수 있다(같은 조 제2항). 임차물의 일부만을 전대하는 것도 전대에 해당한다. 이 사건에서 원고는 피고의 동의 없이 주택의 절반을 전대하였으므로, 피고의 해지 통지는 적법하다.",
     "이 사건 임대차는 피고의 해지 통지로 종료되었다."),
    ("case-HC1", "중고차하자담보", "2026가단99103",
     "원고는 피고로부터 중고차를 매수하였다. 피고는 사고 이력이 없다고 설명하였으나, 원고는 매수 4개월 후 보험 이력 조회를 통해 중대한 사고 이력이 있음을 알게 되었고 계약 해제를 주장하였다.",
     "피고는 원고가 사고 이력을 알 수 있었다고 주장한다. 원고는 피고가 하자를 숨겼다고 주장한다.",
     "매매의 목적물에 하자가 있는 때에는 매수인은 계약을 해제하거나 손해배상을 청구할 수 있다(민법 제580조 제1항, 제575조 제1항 준용). 다만 매수인이 하자가 있는 것을 알았거나 과실로 인하여 이를 알지 못한 때에는 그러하지 아니하다. 중대한 사고 이력을 숨긴 것은 하자에 해당한다. 이 권리는 매수인이 그 사실을 안 날로부터 6개월 내에 행사하여야 한다(민법 제582조). 이 사건에서 원고는 사고 이력을 알지 못하였고 안 날로부터 4개월 만에 권리를 행사하였으므로 해제는 적법하다.",
     "이 사건 매매계약을 해제하고 피고는 대금을 반환한다."),
    ("case-HC2", "불법행위소멸시효", "2026가단99104",
     "원고는 4년 전 A의 교통사고로 다쳤고, 사고 당시 손해와 가해자 A를 알고 있었다. 원고는 지금까지 손해배상 청구권을 행사하지 않았다.",
     "원고는 청구권이 아직 유효하다고 주장한다. 피고는 시효로 소멸하였다고 주장한다.",
     "불법행위로 인한 손해배상청구권은 피해자가 그 손해 및 가해자를 안 날로부터 3년간 행사하지 아니하면 시효로 소멸한다(민법 제766조 제1항). 또한 불법행위를 한 날로부터 10년을 경과하여도 소멸한다(같은 조 제2항). 이 사건에서 원고는 4년 전 사고 당시 손해와 가해자를 알았으므로, 3년의 소멸시효가 완성되었다.",
     "원고의 손해배상청구권은 시효로 소멸하였다."),
    ("case-HX1", "사실적시명예훼손", "2026고단99105",
     "피고는 인터넷 커뮤니티에 이웃 A가 직원들의 임금을 체불하였다는 내용을 게시하였다. 내용은 진실이었으나, 피고의 주된 목적은 A와 개인적 원한이 있어 A의 사정을 지인들에게 알리는 데 있었다.",
     "피고는 진실한 사실이므로 처벌할 수 없다고 주장한다. 검사는 명예훼손죄가 성립한다고 주장한다.",
     "공연히 사실을 적시하여 사람의 명예를 훼손한 자는 그 사실이 진실하더라도 처벌한다(형법 제307조 제1항). 다만 그 행위가 진실한 사실로서 오로지 공공의 이익에 관한 때에는 처벌하지 아니한다(형법 제310조). 오로지 공공의 이익에 관한 것인지는 적시 내용의 공익성, 동기, 피해자와의 관계 등을 종합하여 판단한다. 이 사건에서 적시 내용은 진실이나 주된 동기가 개인적 원한이므로, 오로지 공공의 이익에 관한 것이라고 볼 수 없다.",
     "피고는 명예훼손죄로 유죄이다."),
    ("case-HX2", "업무상횡령", "2026고단99106",
     "피고는 A의 매장에서 매장 자금을 관리하는 직원으로, 매장 금고에 보관 중인 판매 대금 300만 원을 인출하여 개인 생활비에 사용하였다.",
     "피고는 나중에 반환할 의도였다고 주장한다. 검사는 업무상횡령죄가 성립한다고 주장한다.",
     "타인의 재물을 보관하는 자가 그 재물을 횡령한 때에는 횡령죄가 성립한다(형법 제355조 제1항). 업무상의 임무에 위배하여 그 죄를 범한 때에는 업무상횡령죄로 가중처벌한다(형법 제356조). 매장 자금을 관리하는 직원은 타인의 재물을 보관하는 자에 해당하고, 판매 대금을 개인 생활비에 사용한 것은 횡령이다. 나중에 반환할 의도가 있었다고 하여 횡령의 고의가 부정되지 않는다.",
     "피고는 업무상횡령죄로 유죄이다."),
    ("case-HA1", "심판청구기간미고지", "2026구합99107",
     "원고는 5개월 전 구청으로부터 영업정지 처분을 받았다. 처분서에는 행정심판 청구 기간이 기재되어 있지 않았다. 원고는 이제 행정심판을 청구하려고 한다.",
     "피고는 90일의 심판청구 기간이 지났다고 주장한다. 원고는 기간을 알리지 않았으므로 청구할 수 있다고 주장한다.",
     "행정심판은 처분이 있음을 알게 된 날부터 90일 이내에 청구하여야 하고(행정심판법 제27조 제1항), 이 기간은 불변기간이다(같은 조 제4항). 그러나 행정청이 심판청구 기간을 알리지 아니한 경우에는 처분이 있었던 날부터 180일이 지나면 청구하지 못한다(같은 조 제3항, 제6항). 이 사건에서 안 날로부터 90일은 지났으나, 처분서에 심판청구 기간이 기재되어 있지 않았으므로 원고는 처분이 있었던 날부터 180일 이내에 청구할 수 있다. 5개월(약 150일)은 180일 이내이다.",
     "원고의 심판청구는 적법하다."),
    ("case-HA2", "내부검토정보공개", "2026구합99108",
     "원고는 구청 건축심의위원회 회의록의 정보공개를 청구하였다. 구청은 회의록이 내부검토 과정에 있는 사항이라며 비공개하였다. 해당 심의는 2년 전에 종료되어 의결이 확정된 상태였다.",
     "원고는 심의가 종료되었으므로 공개하여야 한다고 주장한다. 피고는 내부검토 사항이므로 비공개할 수 있다고 주장한다.",
     "공공기관이 보유·관리하는 정보는 공개 대상이 되는 것이 원칙이다. 다만 의사결정 과정 또는 내부검토 과정에 있는 사항으로서 공개될 경우 업무의 공정한 수행에 현저한 지장을 초래한다고 인정할 만한 상당한 이유가 있는 정보는 공개하지 아니할 수 있다(공공기관의 정보공개에 관한 법률 제9조 제1항 제5호). 그러나 비공개 대상 정보라도 기간의 경과 등으로 비공개의 필요성이 없어진 경우에는 공개 대상으로 하여야 한다(같은 조 제2항). 이 사건에서 심의는 2년 전에 종료되어 비공개의 필요성이 없어졌으므로, 비공개 처분은 위법하다.",
     "피고의 비공개 처분을 취소한다."),
]

CASES = [
    ("HL1-fixture-purchase", "lease", "건물을 3년간 임차해서 쓰면서 집주인 동의를 받고 시스템 에어컨을 설치했어요. 이제 계약이 끝나서 나가는데, 집주인에게 에어컨을 사 달라고 청구할 수 있나요?", ["law-646"], "case-HL1"),
    ("HL2-unauthorized-sublet", "lease", "임차한 집의 절반을 집주인 동의 없이 지인에게 전대했어요. 집주인이 이 사실을 알고 계약을 해지하겠다고 해요. 정말 해지할 수 있나요?", ["law-629"], "case-HL2"),
    ("HC1-used-car-defect", "civil", "중고차를 산 지 4개월 됐어요. 판매자는 사고가 없다고 했는데 보험 이력을 조회해 보니 중대한 사고 이력이 있었어요. 계약을 해제하고 돈을 돌려받을 수 있나요?", ["law-580", "law-582"], "case-HC1"),
    ("HC2-tort-prescription", "civil", "4년 전 교통사고로 다쳤어요. 사고 당시 가해자가 누구인지는 알고 있었는데 지금까지 손해배상 청구를 안 했어요. 이제라도 청구할 수 있나요?", ["law-766"], "case-HC2"),
    ("HX1-true-defamation", "criminal", "이웃이 임금을 체불했다는 내용을 커뮤니티에 게시했어요. 내용은 사실이에요. 제가 명예훼손으로 처벌받을 수 있나요?", ["law-c307", "law-c310"], "case-HX1"),
    ("HX2-employee-embezzlement", "criminal", "매장 자금을 관리하던 직원이 판매 대금 300만 원을 개인적으로 썼어요. 나중에 갚으려고 했다고는 해요. 어떤 죄가 되나요?", ["law-c355", "law-c356"], "case-HX2"),
    ("HA1-appeal-no-notice", "administrative", "영업정지 처분을 받은 지 5개월 됐어요. 이제 행정심판을 청구하려고 하는데, 처분서에는 심판청구 기간 안내가 없었어요. 아직 가능한가요?", ["law-ap27"], "case-HA1"),
    ("HA2-closed-review-disclosure", "administrative", "건축심의위원회 회의록을 정보공개 청구했는데 구청이 내부검토 사항이라며 비공개했어요. 그 심의는 2년 전에 끝났어요. 이 비공개가 적법한가요?", ["law-id9"], "case-HA2"),
]

COMPARISON = {
    "HL1-fixture-purchase": {
        "core_rule": ["건물 임차인이 사용의 편익을 위하여 임대인의 동의를 얻어 부속한 물건은 임대차 종료 시 매수 청구할 수 있다(민법 646조 1항)",
                       "동의 없이 부속한 물건에는 매수 청구권이 없다"],
        "preparations": ["임대차계약서", "설치에 대한 임대인 동의 기록", "설치 비용·현황 시가 자료"],
        "cautions": ["임대인의 동의가 있었는지가 요건이다", "매수 청구는 임대차 종료 시에 한다"]},
    "HL2-unauthorized-sublet": {
        "core_rule": ["임차인은 임대인의 동의 없이 전대하지 못한다(민법 629조 1항)",
                       "위반 시 임대인은 계약을 해지할 수 있다(629조 2항)", "일부 전대도 전대에 해당한다"],
        "preparations": ["임대차계약서", "전대 사실 기록(전대차 계약·수수료 내역)", "임대인 동의 유무 기록"],
        "cautions": ["동의 유무가 해지 가능 여부를 가른다", "해지는 임대인의 통지가 있어야 한다"]},
    "HC1-used-car-defect": {
        "core_rule": ["매매 목적물에 하자가 있으면 매수인은 해제 또는 손해배상을 청구할 수 있다(민법 580조 1항)",
                       "매수인이 하자를 알았거나 과실로 알지 못한 때에는 제외된다",
                       "권리는 사실을 안 날로부터 6개월 내에 행사해야 한다(민법 582조)"],
        "preparations": ["매매계약서", "판매자의 무사고 설명 기록", "사고 이력 조회 결과", "하자를 안 시점 자료"],
        "cautions": ["6개월의 권리행사기간이 있다", "매수인의 악의·과실 여부가 결론을 가른다"]},
    "HC2-tort-prescription": {
        "core_rule": ["불법행위 손해배상청구권은 손해 및 가해자를 안 날로부터 3년간 행사하지 않으면 시효로 소멸한다(민법 766조 1항)",
                       "불법행위를 한 날로부터 10년을 경과하여도 소멸한다(766조 2항)"],
        "preparations": ["사고·손해 발생 기록", "손해와 가해자를 안 시점 자료"],
        "cautions": ["3년은 사고일이 아니라 손해와 가해자를 안 날부터 기산한다", "3년과 10년 두 기간이 함께 적용된다"]},
    "HX1-true-defamation": {
        "core_rule": ["공연히 사실을 적시하여 명예를 훼손하면 진실한 사실이라도 처벌한다(형법 307조 1항)",
                       "진실한 사실로서 오로지 공공의 이익에 관한 때에는 처벌하지 않는다(형법 310조)",
                       "공공의 이익 해당성은 내용·동기·관계 등을 종합하여 판단한다"],
        "preparations": ["게시물 내용·게시 시점 기록", "적시 사실의 진실성 자료", "게시 동기·경위 기록"],
        "cautions": ["진실한 사실도 명예훼손이 될 수 있다", "'오로지' 공공의 이익이어야 면책된다"]},
    "HX2-employee-embezzlement": {
        "core_rule": ["타인의 재물을 보관하는 자가 횡령하면 횡령죄가 성립한다(형법 355조 1항)",
                       "업무상의 임무에 위배하여 범하면 업무상횡령으로 가중처벌한다(형법 356조)",
                       "나중에 반환할 의도가 있어도 횡령의 고의는 부정되지 않는다"],
        "preparations": ["자금 관리 업무 분장 기록", "자금 인출·사용 내역(계좌 내역 등)", "금액·시점 기록"],
        "cautions": ["'타인의 재물을 보관하는 자'에 해당하는지가 요건이다", "업무상 임무 위배 여부가 가중처벌을 가른다"]},
    "HA1-appeal-no-notice": {
        "core_rule": ["행정심판은 처분이 있음을 안 날부터 90일 이내에 청구해야 한다(행정심판법 27조 1항)",
                       "90일 기간은 불변기간이다(27조 4항)",
                       "행정청이 청구 기간을 알리지 않은 경우 처분이 있은 날부터 180일 이내에 청구할 수 있다(27조 6항, 3항)"],
        "preparations": ["처분서(심판청구 기간 기재 여부 확인)", "처분 수령·인지 시점 자료"],
        "cautions": ["기간 고지 여부에 따라 적용 기간이 달라진다", "180일이 지나면 청구할 수 없다"]},
    "HA2-closed-review-disclosure": {
        "core_rule": ["공공기관 보유 정보는 공개가 원칙이다(정보공개법 9조 1항)",
                       "내부검토 과정 사항으로서 공개 시 업무의 공정한 수행에 현저한 지장을 초래할 상당한 이유가 있으면 비공개할 수 있다(9조 1항 5호)",
                       "기간 경과 등으로 비공개 필요성이 없어지면 공개 대상으로 해야 한다(9조 2항)"],
        "preparations": ["정보공개 청구서·비공개 처분서", "심의 종료·의결 확정 시점 자료"],
        "cautions": ["내부검토 사항이라고 무조건 비공개는 아니다", "심의 종료·시간 경과로 공개 의무가 생길 수 있다"]},
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {law for _, law, _ in STATUTES}
    laws = {}
    for path in sorted(CORPUS.glob("*.json")):
        data = json.loads(path.read_text())
        if data["law_name"] in wanted and data["law_name"] not in laws:
            laws[data["law_name"]] = (data, path.name)
    missing = wanted - laws.keys()
    if missing:
        raise SystemExit(f"laws not found: {missing}")
    sources = []
    for sid, law, number in STATUTES:
        data, corpus_file = laws[law]
        article = next(a for a in data["articles"] if a["number"] == number)
        quote = article_text(article)
        fname = f"{sid}.txt"
        (OUT / fname).write_text(quote + "\n")
        sources.append({"id": sid, "kind": "statute", "law_name": law, "article": number,
                        "effective_date": data["effective_date"], "quote": quote,
                        "extraction_file": fname, "sha256": hashlib.sha256((quote + "\n").encode()).hexdigest(),
                        "url": data["source_url"], "corpus_file": f"data/api_laws/statute-api-full/{corpus_file}",
                        "method": "official law.go.kr API corpus (statute-api-full-v1), article extracted verbatim"})
    for cid, title, number, facts, claims, holding, order in JUDGMENTS:
        content = (f"{FICTIONAL_MARK}\n가상지방법원 판결\n사건 {number} {title}\n"
                   f"1. 기초사실\n{facts}\n2. 당사자의 주장\n{claims}\n3. 판단\n{holding}\n4. 결론\n{order}\n")
        fname = f"{cid}.txt"
        (OUT / fname).write_text(content)
        assert holding in content
        sources.append({"id": cid, "kind": "case", "title": title, "court": "가상지방법원", "case_number": number,
                        "quote": holding, "extraction_file": fname,
                        "sha256": hashlib.sha256(content.encode()).hexdigest()})
    cases = [{"id": cid, "domain": domain, "message": message, "sources": [*statutes, case_id], "case_source": case_id}
             for cid, domain, message, statutes, case_id in CASES]
    assert set(COMPARISON) == {c["id"] for c in cases}
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(COMPARISON, ensure_ascii=False, indent=2) + "\n")
    print(f"built {len(cases)} held-out cases, {len(sources)} sources -> {OUT}")


if __name__ == "__main__":
    main()
