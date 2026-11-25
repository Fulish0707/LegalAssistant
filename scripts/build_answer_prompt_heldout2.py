"""Build the SECOND held-out check inputs (data/eval/answer_prompt_heldout2).

Eight new cases for the refined preparations/cautions guidance (anti-fabrication).
New issues and statutes, none reused from answer_generation or answer_prompt_heldout.
Comparison table fixed before any model run; adoption gates in README.md.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src .venv/bin/python scripts/build_answer_prompt_heldout2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_answer_eval import FICTIONAL_MARK, article_text  # noqa: E402

OUT = ROOT / "data/eval/answer_prompt_heldout2"
CORPUS = ROOT / "data/api_laws/statute-api-full"

STATUTES = [
    ("law-640", "민법", "제640조"), ("law-h4", "주택임대차보호법", "제4조"),
    ("law-548", "민법", "제548조"), ("law-396", "민법", "제396조"), ("law-763", "민법", "제763조"),
    ("law-c257", "형법", "제257조"), ("law-c283", "형법", "제283조"),
    ("law-ap21", "행정절차법", "제21조"), ("law-ap22", "행정절차법", "제22조"),
    ("law-at30", "행정심판법", "제30조"),
]

JUDGMENTS = [
    ("case-TL1", "차임연체해지", "2026가단99201",
     "원고는 피고 소유 건물을 월 차임 50만 원에 임차하였다. 원고가 차임 2기분 100만 원을 연체하자 피고가 임대차 해지를 통지하였는데, 원고는 해지 통지 전에 연체액 중 30만 원을 지급한 상태였다.",
     "피고는 2기 연체이므로 해지할 수 있다고 주장한다. 원고는 일부 변제로 연체액이 2기에 미달한다고 주장한다.",
     "건물의 임대차에서 임차인의 차임연체액이 2기의 차임액에 달하는 때에는 임대인은 계약을 해지할 수 있다(민법 제640조). 연체액은 해지 통지 시를 기준으로 산정하고, 통지 전에 지급된 금액은 공제한다. 이 사건에서 원고가 통지 전에 30만 원을 지급하였으므로 통지 시 연체액은 70만 원으로 2기(100만 원)에 미달한다. 따라서 피고의 해지 통지는 효력이 없다.",
     "피고의 해지 통지는 효력이 없다."),
    ("case-TL2", "임대차기간2년", "2026가단99202",
     "원고는 피고 소유 주택을 계약기간 1년으로 정하여 임차하였다. 1년이 지나자 피고는 명도를 요구하였고, 원고는 기간이 2년이라고 주장하였다.",
     "피고는 약정한 1년 기간이 유효하다고 주장한다. 원고는 법률에 따라 2년으로 본다고 주장한다.",
     "기간을 정하지 아니하거나 2년 미만으로 정한 주택임대차는 그 기간을 2년으로 본다(주택임대차보호법 제4조 제1항). 다만 임차인은 2년 미만으로 정한 기간이 유효함을 주장할 수 있다(같은 항 단서). 이는 임차인 보호를 위한 것이므로 임대인이 약정한 짧은 기간을 임차인에게 주장할 수는 없다. 이 사건 계약기간은 1년으로 정해졌으므로 원고는 2년의 기간을 주장할 수 있다.",
     "이 사건 임대차기간은 2년으로 보고, 피고의 명도 요구는 이유 없다."),
    ("case-TC1", "해제원상회복", "2026가단99203",
     "원고는 피고에게 매매대금 500만 원을 지급하였으나, 피고의 채무불이행으로 계약을 적법하게 해제하였다. 원고는 대금 반환과 이자를 구하였다.",
     "피고는 원금만 반환하면 된다고 주장한다. 원고는 받은 날로부터의 이자를 더하여야 한다고 주장한다.",
     "당사자 일방이 계약을 해제한 때에는 각 당사자는 상대방에 대하여 원상회복의 의무가 있다(민법 제548조 제1항). 반환할 금전에는 그 받은 날로부터 이자를 가하여야 한다(같은 조 제2항). 따라서 피고는 매매대금 500만 원에 그 받은 날로부터의 이자를 더하여 반환하여야 한다.",
     "피고는 원고에게 500만 원과 그 받은 날로부터의 이자를 반환한다."),
    ("case-TC2", "과실상계", "2026가단99204",
     "원고는 피고의 과실로 발생한 교통사고로 다쳤다. 다만 원고에게도 무단횡단 과실이 30% 인정되었다. 원고는 전액 배상을 구하였다.",
     "원고는 전액 배상을 구한다. 피고는 원고의 과실을 참작하여 배상액을 정하여야 한다고 주장한다.",
     "채무불이행에 관하여 채권자에게 과실이 있는 때에는 법원은 손해배상의 책임 및 그 금액을 정함에 이를 참작하여야 한다(민법 제396조). 이 규정은 불법행위로 인한 손해배상에 준용된다(민법 제763조). 이 사건에서 원고에게도 30%의 과실이 있으므로, 배상액은 이를 참작하여 정한다.",
     "피고의 배상액은 원고의 과실을 참작한 범위에서 인정된다."),
    ("case-TX1", "상해죄", "2026고단99205",
     "피고는 말다툼 중 A를 밀었고, A는 넘어져 전치 2주의 상해를 입었다.",
     "피고는 상해의 고의가 없었다고 주장한다. 검사는 상해죄가 성립한다고 주장한다.",
     "사람의 신체를 상해한 자는 7년 이하의 징역, 10년 이하의 자격정지 또는 1천만 원 이하의 벌금에 처한다(형법 제257조 제1항). 상해죄는 폭행의 인식과 상해 결과의 발생으로 성립하고, 특정한 상해의 고의까지 요구하지 않는다. 이 사건에서 피고가 A를 밀어 전치 2주의 상해 결과가 발생하였으므로 상해죄가 성립한다.",
     "피고는 상해죄로 유죄이다."),
    ("case-TX2", "협박죄", "2026고단99206",
     "피고는 채권자 A에게 '계속 독촉하면 가만두지 않겠다'고 말하여 A에게 공포심을 일으켰다. A는 피고의 처벌을 원하지 않는다는 의사를 밝혔다.",
     "피고는 감정적 표현에 불과하다고 주장한다. 검사는 협박죄가 성립한다고 주장한다.",
     "사람을 협박한 자는 3년 이하의 징역, 500만 원 이하의 벌금, 구류 또는 과료에 처한다(형법 제283조 제1항). 협박은 사람에게 공포심을 일으킬 정도의 해악을 고지하는 것이다. 협박죄는 피해자의 명시한 의사에 반하여 공소를 제기할 수 없다(같은 조 제3항). 이 사건에서 피고의 말은 공포심을 일으킬 정도의 해악 고지에 해당하여 협박죄가 성립하나, 피해자 A가 처벌을 원하지 않으므로 공소를 제기할 수 없다.",
     "협박죄는 성립하나 피해자의 명시한 의사에 반하여 공소 제기할 수 없다."),
    ("case-TA1", "사전통지누락", "2026구합99207",
     "피고 구청은 원고에게 영업정지 2개월 처분을 하면서 사전 통지와 의견제출 기회를 주지 않았다. 공공의 안전을 위한 긴급한 필요가 있는 경우는 아니었다.",
     "원고는 절차적 위법이 있다고 주장한다. 피고는 위반 사실이 명백하여 의견청취가 필요 없었다고 주장한다.",
     "행정청은 당사자에게 의무를 부과하거나 권익을 제한하는 처분을 하는 경우 처분의 제목, 원인 사실과 내용 및 법적 근거, 의견제출의 뜻과 기한 등을 미리 통지하여야 한다(행정절차법 제21조 제1항). 청문·공청회 경우 외에는 당사자에게 의견제출의 기회를 주어야 한다(제22조 제3항). 공공의 안전 또는 복리를 위하여 긴급히 처분할 필요가 있는 경우 등에는 사전 통지를 하지 아니할 수 있으나(제21조 제4항), 이 사건에는 그러한 사유가 없다. 따라서 이 사건 처분은 절차적 하자가 있다.",
     "이 사건 처분은 절차적 하자로 위법하다."),
    ("case-TA2", "심판집행정지", "2026구합99208",
     "원고는 과징금 3,000만 원 부과 처분을 받고 행정심판을 청구하였다. 행정청은 다음 주 강제징수를 예고하였다. 원고는 징수되면 직원 급여 지급 자금이 소진되어 영업에 중대한 손해가 생긴다며 집행정지를 신청하였다.",
     "원고는 중대한 손해 예방의 긴급한 필요가 있다고 주장한다. 행정청은 심판청구가 집행을 정지시키지 않는다고 주장한다.",
     "심판청구는 처분의 효력이나 그 집행 또는 절차의 속행에 영향을 주지 아니한다(행정심판법 제30조 제1항). 그러나 처분이나 그 집행 때문에 중대한 손해가 생기는 것을 예방할 필요성이 긴급하다고 인정할 때에는 위원회는 직권 또는 당사자의 신청으로 집행정지를 결정할 수 있다(같은 조 제2항). 집행정지는 공공복리에 중대한 영향을 미칠 우려가 있을 때에는 허용되지 아니한다(같은 조 제3항). 이 사건에서 과징금이 징수되면 직원 급여 자금이 소진되어 영업에 중대한 손해가 생기고 다음 주 강제징수가 예정되어 있어 긴급한 필요가 인정되며, 공공복리에 대한 중대한 영향 우려는 없다.",
     "재결 시까지 과징금 부과 처분의 집행을 정지한다."),
]

CASES = [
    ("TL1-rent-arrears", "lease", "월세를 두 달 밀려서 집주인이 계약을 해지한다고 해요. 그런데 해지 통지 전에 밀린 돈 중 30만 원을 냈어요. 그래도 해지할 수 있나요?", ["law-640"], "case-TL1"),
    ("TL2-one-year-term", "lease", "집을 계약기간 1년으로 계약했는데 1년이 지났다고 나가라고 해요. 2년은 살 수 없는 건가요?", ["law-h4"], "case-TL2"),
    ("TC1-rescission-refund", "civil", "상대방이 계약을 이행하지 않아서 적법하게 해제했어요. 이미 낸 돈 500만 원을 돌려받을 수 있나요? 이자도 받을 수 있는지 알고 싶어요.", ["law-548"], "case-TC1"),
    ("TC2-comparative-fault", "civil", "교통사고로 다쳤는데 조사에서 제 무단횡단 과실도 30% 있다고 해요. 배상액이 줄어들나요?", ["law-396", "law-763"], "case-TC2"),
    ("TX1-push-injury", "criminal", "말다툼 중에 상대를 밀쳤는데 넘어져서 전치 2주가 나왔어요. 어느 정도의 처벌을 받을 수 있나요?", ["law-c257"], "case-TX1"),
    ("TX2-debtor-threat", "criminal", "채무자가 '계속 독촉하면 가만두지 않겠다'고 했어요. 협박죄가 되나요? 저는 사실 처벌을 원하지는 않아요.", ["law-c283"], "case-TX2"),
    ("TA1-no-prior-notice", "administrative", "구청이 사전 통지 없이 영업정지 처분을 했어요. 긴급한 사정은 없었어요. 절차상 문제가 있나요?", ["law-ap21", "law-ap22"], "case-TA1"),
    ("TA2-appeal-stay", "administrative", "과징금 3,000만 원 처분을 받고 행정심판을 청구했어요. 행정청은 다음 주에 강제징수를 하겠다고 해요. 징수되면 직원 월급도 줄 수 없어요. 심판으로 집행을 멈출 수 있나요?", ["law-at30"], "case-TA2"),
]

COMPARISON = {
    "TL1-rent-arrears": {
        "core_rule": ["차임연체액이 2기의 차임액에 달하면 임대인은 계약을 해지할 수 있다(민법 640조)",
                       "연체액은 해지 통지 시 기준으로 산정하고 통지 전 지급액은 공제한다"],
        "preparations": ["임대차계약서(차임 액수)", "차임 지급 내역", "해지 통지 시점·내용 기록"],
        "cautions": ["통지 전 일부 변제로 연체액이 2기 미만이 되면 해지할 수 없다", "연체액이 2기에 달하는지가 기준이다"]},
    "TL2-one-year-term": {
        "core_rule": ["기간을 정하지 않거나 2년 미만으로 정한 주택임대차는 기간을 2년으로 본다(주택임대차보호법 4조 1항)",
                       "임차인은 2년 미만으로 정한 기간이 유효함을 주장할 수 있다(같은 항 단서)",
                       "임대인이 약정한 짧은 기간을 임차인에게 주장할 수는 없다"],
        "preparations": ["임대차계약서(기간 조항)"],
        "cautions": ["짧은 기간을 주장할 선택권은 임차인에게 있다", "기간이 끝나도 보증금을 반환받을 때까지 임대차관계가 존속되는 것으로 본다(4조 2항)"]},
    "TC1-rescission-refund": {
        "core_rule": ["계약 해제 시 각 당사자는 상대방에게 원상회복 의무가 있다(민법 548조 1항)",
                       "반환할 금전에는 받은 날로부터 이자를 가한다(548조 2항)",
                       "원상회복은 제삼자의 권리를 해하지 못한다(548조 1항 단서)"],
        "preparations": ["계약서·대금 지급 기록", "해제 의사표시 기록"],
        "cautions": ["이자는 해제일이 아니라 받은 날부터 기산한다", "제삼자의 권리를 해할 수 없다"]},
    "TC2-comparative-fault": {
        "core_rule": ["채권자에게 과실이 있으면 법원은 손해배상 책임과 금액을 정함에 이를 참작하여야 한다(민법 396조)",
                       "이 규정은 불법행위 손해배상에 준용된다(민법 763조)"],
        "preparations": ["쌍방 과실을 나타내는 사고 기록(CCTV·경찰 조사 기록)", "손해 내역 자료"],
        "cautions": ["과실상계는 참작하여야 하는 사항이다(임의 아님)", "피해자 과실 비율이 배상액을 줄인다"]},
    "TX1-push-injury": {
        "core_rule": ["사람의 신체를 상해한 자는 7년 이하 징역, 10년 이하 자격정지 또는 1천만 원 이하 벌금에 처한다(형법 257조 1항)",
                       "상해죄는 폭행의 인식과 상해 결과의 발생으로 성립한다"],
        "preparations": ["진단서(상해 정도·전치 기간)", "사건 경위 기록(CCTV·진술)"],
        "cautions": ["상해의 정도가 양형에 영향을 준다", "직계존속에 대한 범죄는 법정형이 다르다(257조 2항)"]},
    "TX2-debtor-threat": {
        "core_rule": ["사람을 협박한 자는 3년 이하 징역, 500만 원 이하 벌금, 구류 또는 과료에 처한다(형법 283조 1항)",
                       "협박은 공포심을 일으킬 정도의 해악 고지이다",
                       "협박죄는 피해자의 명시한 의사에 반하여 공소를 제기할 수 없다(283조 3항)"],
        "preparations": ["협박 발언 기록(문자·녹취)", "피해자의 처벌 의사 기록"],
        "cautions": ["피해자가 처벌을 원하지 않으면 공소 제기할 수 없다", "공포심을 일으킬 정도의 해악 고지인지가 기준이다"]},
    "TA1-no-prior-notice": {
        "core_rule": ["의무 부과·권익 제한 처분 시 처분 제목·원인 사실·내용·법적 근거·의견제출 안내 등을 미리 통지해야 한다(행정절차법 21조 1항)",
                       "청문·공청회 경우 외에는 의견제출 기회를 주어야 한다(22조 3항)",
                       "공공의 안전·복리를 위한 긴급한 경우 등에는 사전 통지를 생략할 수 있다(21조 4항)"],
        "preparations": ["처분서", "사전 통지·의견제출 기회 수령 여부 기록"],
        "cautions": ["생략 사유는 법령이 정한 경우로 제한된다", "사전 통지를 생략하면 그 사유를 알려야 한다(21조 6항)"]},
    "TA2-appeal-stay": {
        "core_rule": ["심판청구는 처분의 효력·집행에 영향을 주지 않는다(행정심판법 30조 1항)",
                       "중대한 손해 예방의 긴급한 필요가 있으면 위원회가 직권 또는 신청으로 집행정지를 결정할 수 있다(30조 2항)",
                       "공공복리에 중대한 영향을 미칠 우려가 있으면 집행정지는 허용되지 않는다(30조 3항)"],
        "preparations": ["처분서", "심판청구 기록", "중대한 손해·긴급성 자료(자금 사정·급여 지급 내역 등)"],
        "cautions": ["집행정지는 별도 신청과 위원회 결정이 있어야 한다(자동 정지 아님)", "공공복리 예외가 있다"]},
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
    print(f"built {len(cases)} held-out-2 cases, {len(sources)} sources -> {OUT}")


if __name__ == "__main__":
    main()
