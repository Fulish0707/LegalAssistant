"""Build the held-out set for the consulting adapter v2 decision (data/eval/adapter_consult_heldout2).

Eight new cases; statutes not used in any earlier answer set (checked). Comparison table fixed before any
model run; gates in README.md.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src .venv/bin/python scripts/build_adapter_heldout2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_answer_eval import FICTIONAL_MARK, article_text  # noqa: E402

OUT = ROOT / "data/eval/adapter_consult_heldout2"
CORPUS = ROOT / "data/api_laws/statute-api-full"
PREVIOUS = ["answer_generation", "answer_prompt_heldout", "answer_prompt_heldout2", "adapter_consult_heldout"]

STATUTES = [
    ("law-h3-3", "주택임대차보호법", "제3조의3"), ("law-627", "민법", "제627조"),
    ("law-536", "민법", "제536조"), ("law-391", "민법", "제391조"),
    ("law-c260", "형법", "제260조"), ("law-c366", "형법", "제366조"),
    ("law-oi11", "공공기관의 정보공개에 관한 법률", "제11조"), ("law-ap24", "행정절차법", "제24조"),
]

JUDGMENTS = [
    ("case-WL1", "임차권등기명령", "2026카임99401",
     "신청인은 주택 임대차가 기간 만료로 끝났으나 임대인으로부터 보증금 1억 원을 돌려받지 못하였다. 신청인은 직장 때문에 다른 곳으로 이사해야 하는 상황에서 임차권등기명령을 신청하였다.",
     "임대인은 보증금을 마련할 때까지 기다려 달라고 주장한다. 신청인은 이사하더라도 보증금을 돌려받을 권리를 지키고자 한다.",
     "임대차가 끝난 후 보증금이 반환되지 아니한 경우 임차인은 임차주택 소재지 관할 법원에 임차권등기명령을 신청할 수 있다(주택임대차보호법 제3조의3 제1항). 신청서에는 신청의 이유와 임차권등기의 원인이 된 사실을 소명하여야 한다(같은 조 제2항). 임차권등기를 마치면 대항력과 우선변제권을 취득하고, 이미 취득한 대항력이나 우선변제권은 그대로 유지되며, 임차권등기 이후에는 대항요건을 상실하더라도 이를 잃지 않는다(같은 조 제5항). 따라서 임차인은 임차권등기를 마친 뒤에 이사하여야 기존 권리를 지킬 수 있다. 임차인은 신청과 등기에 든 비용을 임대인에게 청구할 수 있다(같은 조 제8항). 신청인은 임대차 종료와 보증금 미반환을 소명하였으므로 신청은 이유 있다.",
     "임차권등기를 명한다."),
    ("case-WL2", "일부멸실차임감액", "2026가단99402",
     "원고는 방 두 개짜리 주택을 월 차임 80만 원에 임차하였다. 윗집 누수로 방 하나의 천장이 무너져 4개월째 사용하지 못하고 있으며, 원고에게는 잘못이 없다. 원고는 차임 감액을 청구하였다.",
     "피고는 주택 전체를 쓸 수 없는 것은 아니므로 차임을 감액할 이유가 없다고 주장한다. 원고는 사용하지 못하는 부분만큼 감액하여야 한다고 주장한다.",
     "임차물의 일부가 임차인의 과실 없이 멸실 기타 사유로 사용, 수익할 수 없는 때에는 임차인은 그 부분의 비율에 의한 차임의 감액을 청구할 수 있다(민법 제627조 제1항). 이 사건에서 원고의 과실 없이 방 하나를 사용할 수 없게 되었으므로, 원고는 사용할 수 없는 부분의 비율에 따라 차임 감액을 청구할 수 있다. 다만 남은 부분으로 임차의 목적을 달성할 수 없는 때에는 임차인은 계약을 해지할 수도 있다(같은 조 제2항).",
     "차임을 사용할 수 없는 부분의 비율에 따라 감액한다."),
    ("case-WC1", "동시이행항변", "2026가단99403",
     "원고는 피고로부터 중고 자동차를 사기로 하고 계약금을 지급하였다. 잔금 지급일과 차량 인도일은 같은 날로 정하였다. 그날 피고는 차량을 인도할 준비를 하지 않은 채 잔금부터 지급하라고 요구하였다.",
     "피고는 원고가 잔금을 먼저 지급하여야 한다고 주장한다. 원고는 차량 인도와 동시에 지급하겠다고 주장한다.",
     "쌍무계약의 당사자 일방은 상대방이 그 채무이행을 제공할 때까지 자기의 채무이행을 거절할 수 있다(민법 제536조 제1항 본문). 다만 상대방의 채무가 변제기에 있지 아니한 때에는 그러하지 아니하다(같은 항 단서). 먼저 이행하여야 할 당사자라도 상대방의 이행이 곤란할 현저한 사유가 있으면 이행을 거절할 수 있다(같은 조 제2항). 이 사건에서 잔금 지급과 차량 인도는 같은 날 이행하기로 하였으므로 동시이행 관계에 있고, 피고가 차량 인도를 제공하지 않은 이상 원고는 잔금 지급을 거절할 수 있다.",
     "원고는 피고의 차량 인도와 동시에 잔금을 지급할 의무가 있다."),
    ("case-WC2", "이사파손", "2026가단99404",
     "원고는 피고 이사업체와 이사 계약을 맺었다. 피고가 보낸 작업자가 운반 중 부주의로 원고의 냉장고를 떨어뜨려 고장 냈다. 피고는 작업자 개인의 잘못이라며 책임을 부인하였다.",
     "피고는 작업자 개인이 배상하여야 한다고 주장한다. 원고는 피고가 계약상 책임을 진다고 주장한다.",
     "채무자가 타인을 사용하여 이행하는 경우에는 피용자의 고의나 과실은 채무자의 고의나 과실로 본다(민법 제391조). 이사 운반은 피고가 계약상 부담한 채무이고, 피고는 작업자를 사용하여 이를 이행하였다. 작업자의 부주의는 피고의 과실로 보므로, 피고는 채무불이행으로 인한 손해를 배상할 책임이 있다. 손해는 파손과 그 손해액으로 증명되어야 한다.",
     "피고는 원고에게 냉장고 수리비 상당의 손해를 배상한다."),
    ("case-WX1", "단순폭행", "2026고단99405",
     "피고인은 술자리에서 시비 끝에 피해자의 뺨을 한 차례 때렸다. 피해자는 다치지 않았다. 피해자는 수사 단계에서 처벌을 원한다는 의사를 밝혔다.",
     "피고인은 가벼운 접촉이었다고 주장한다. 검사는 폭행죄가 성립한다고 주장한다.",
     "사람의 신체에 대하여 폭행을 가한 자는 2년 이하의 징역, 500만 원 이하의 벌금, 구류 또는 과료에 처한다(형법 제260조 제1항). 뺨을 때린 것은 신체에 대한 유형력의 행사로 폭행에 해당한다. 상해에 이르지 않았으므로 폭행죄가 문제 되고, 상처가 생겼다면 다른 죄가 문제 될 수 있다. 폭행죄는 피해자의 명시한 의사에 반하여 공소를 제기할 수 없는데(같은 조 제3항), 피해자는 처벌을 원한다는 의사를 밝혔다. 따라서 폭행죄가 성립한다.",
     "피고인은 폭행죄로 유죄이다."),
    ("case-WX2", "사이드미러손괴", "2026고단99406",
     "피고인은 주차 문제로 다투다가 화가 나 피해자 차량의 사이드미러를 발로 차 부쉈다. 블랙박스 영상에 그 장면이 찍혔다.",
     "피고인은 실수로 부딪쳤다고 주장한다. 검사는 고의로 손괴하였다고 주장한다.",
     "타인의 재물을 손괴 또는 은닉 기타 방법으로 그 효용을 해한 자는 3년 이하의 징역 또는 700만 원 이하의 벌금에 처한다(형법 제366조). 재물손괴죄는 고의가 있어야 성립하고, 실수로 부딪쳐 망가뜨린 경우에는 이 죄로 처벌되지 않는다. 블랙박스 영상에 따르면 피고인이 화가 나 사이드미러를 발로 찬 사실이 인정되므로 고의가 있다. 따라서 재물손괴죄가 성립한다.",
     "피고인은 재물손괴죄로 유죄이다."),
    ("case-WA1", "정보공개결정지연", "2026구합99407",
     "원고는 피고 구청에 정보공개를 청구하였다. 피고는 청구를 받은 날부터 20일이 지나도록 공개 여부를 결정하지 않았고, 결정기간 연장을 통지하지도 않았다.",
     "피고는 업무가 많아 늦어졌다고 주장한다. 원고는 법정 기간을 어겼다고 주장한다.",
     "공공기관은 정보공개 청구를 받으면 그 청구를 받은 날부터 10일 이내에 공개 여부를 결정하여야 한다(공공기관의 정보공개에 관한 법률 제11조 제1항). 부득이한 사유로 그 기간 이내에 결정할 수 없을 때에는 10일의 범위에서 결정기간을 연장할 수 있으나, 연장된 사실과 연장 사유를 청구인에게 지체 없이 문서로 통지하여야 한다(같은 조 제2항). 피고는 연장 통지 없이 10일을 넘겼으므로 법정 결정기간을 지키지 않았다. 청구된 정보가 제3자와 관련이 있으면 그 사실을 제3자에게 통지하여야 하고(같은 조 제3항), 다른 공공기관이 보유한 정보이면 소관 기관으로 이송하고 청구인에게 통지하여야 한다(같은 조 제4항).",
     "피고는 법정 기간 내에 공개 여부를 결정하지 않았다."),
    ("case-WA2", "구두영업정지", "2026구합99408",
     "피고 구청 직원은 원고에게 전화로 다음 날부터 영업을 정지하라고 말하였을 뿐 처분 문서를 주지 않았다. 공공의 안전을 위해 긴급히 처분할 사정은 없었고, 영업정지는 경미한 사안이 아니었다.",
     "피고는 전화로 충분히 알렸다고 주장한다. 원고는 문서 없는 처분은 위법하다고 주장한다.",
     "행정청이 처분을 할 때에는 다른 법령등에 특별한 규정이 있는 경우를 제외하고는 문서로 하여야 한다(행정절차법 제24조 제1항). 공공의 안전 또는 복리를 위하여 긴급히 처분을 할 필요가 있거나 사안이 경미한 경우에는 말이나 전화 등으로 처분할 수 있으나, 당사자가 요청하면 지체 없이 처분에 관한 문서를 주어야 한다(같은 조 제2항). 이 사건 영업정지는 경미한 사안이 아니고 긴급한 사정도 없었으므로 문서로 하여야 하는데, 전화로만 통보하였으므로 처분의 방식에 하자가 있다.",
     "이 사건 처분은 방식의 하자로 위법하다."),
]

CASES = [
    ("WL1-lease-registration", "lease", "전세 계약이 끝났는데 집주인이 보증금 1억 원을 안 돌려줘요. 직장 때문에 다음 달에 이사를 가야 하는데, 이사 가면 보증금을 받을 권리가 없어지나요?", ["law-h3-3"], "case-WL1"),
    ("WL2-partial-loss-rent", "lease", "월세 80만 원에 방 두 개짜리 집을 빌렸는데, 윗집 누수로 방 하나 천장이 무너져서 몇 달째 못 쓰고 있어요. 제 잘못은 없어요. 월세를 깎아 달라고 할 수 있나요?", ["law-627"], "case-WL2"),
    ("WC1-simultaneous-performance", "civil", "중고차를 사기로 하고 계약금을 냈어요. 잔금 날짜랑 차를 넘겨받는 날을 같은 날로 정했는데, 판매자가 차는 준비도 안 해 놓고 잔금부터 내라고 해요. 제가 먼저 잔금을 내야 하나요?", ["law-536"], "case-WC1"),
    ("WC2-moving-damage", "civil", "이사업체에 이사를 맡겼는데 업체가 보낸 작업자가 제 냉장고를 떨어뜨려서 고장 났어요. 업체는 작업자 개인 잘못이라 책임이 없대요. 업체에 배상을 받을 수 있나요?", ["law-391"], "case-WC2"),
    ("WX1-slap-assault", "criminal", "술자리에서 시비가 붙었는데 상대가 제 뺨을 때렸어요. 다친 데는 없어요. 처벌받게 할 수 있나요?", ["law-c260"], "case-WX1"),
    ("WX2-mirror-damage", "criminal", "옆집 사람이 주차 문제로 다투다가 제 차 사이드미러를 발로 차서 부쉈어요. 처벌받게 할 수 있나요?", ["law-c366"], "case-WX2"),
    ("WA1-disclosure-delay", "administrative", "구청에 정보공개를 청구했는데 2주가 넘도록 아무 연락이 없어요. 원래 언제까지 답해야 하나요?", ["law-oi11"], "case-WA1"),
    ("WA2-phone-suspension", "administrative", "구청 직원이 전화로 '내일부터 영업정지'라고만 했어요. 서류는 하나도 못 받았어요. 이것도 효력이 있나요?", ["law-ap24"], "case-WA2"),
]

COMPARISON = {
    "WL1-lease-registration": {
        "core_rule": ["임대차가 끝난 후 보증금을 돌려받지 못하면 임차권등기명령을 신청할 수 있다(주택임대차보호법 3조의3 1항)",
                      "임차권등기를 마치면 대항력·우선변제권을 취득하거나 유지하고, 그 뒤에는 대항요건을 잃어도 권리를 잃지 않는다(3조의3 5항)"],
        "preparations": ["임대차계약서", "보증금 지급 기록", "임대차 종료를 보여 주는 자료(계약기간·종료 통지)", "주민등록·확정일자 등 대항력·우선변제권 취득 자료"],
        "cautions": ["임차권등기를 마친 뒤에 이사해야 기존 권리를 지킬 수 있다", "신청과 등기 비용은 임대인에게 청구할 수 있다(3조의3 8항)",
                     "신청 이유와 등기 원인 사실을 소명해야 한다(3조의3 2항)"]},
    "WL2-partial-loss-rent": {
        "core_rule": ["임차인 과실 없이 임차물 일부를 사용·수익할 수 없으면 그 부분 비율로 차임 감액을 청구할 수 있다(민법 627조 1항)",
                      "남은 부분으로 임차 목적을 달성할 수 없으면 계약을 해지할 수 있다(627조 2항)"],
        "preparations": ["사용할 수 없는 부분의 사진", "사용 불가 부분의 면적·기간을 보여 주는 자료", "임대인에게 알린·감액을 요청한 기록"],
        "cautions": ["임차인에게 과실이 있으면 감액을 청구할 수 없다", "감액은 사용할 수 없는 부분의 비율에 따른다",
                     "남은 부분으로 목적을 달성할 수 없으면 해지도 가능하다"]},
    "WC1-simultaneous-performance": {
        "core_rule": ["쌍무계약의 당사자는 상대방이 이행을 제공할 때까지 자기 이행을 거절할 수 있다(민법 536조 1항)",
                      "상대방 채무가 변제기에 있지 않으면 거절할 수 없다(536조 1항 단서)",
                      "먼저 이행할 당사자도 상대방 이행이 곤란할 현저한 사유가 있으면 거절할 수 있다(536조 2항)"],
        "preparations": ["매매계약서(잔금일·인도일)", "계약금 송금 기록", "판매자와 주고받은 메시지"],
        "cautions": ["계약상 먼저 이행하기로 했다면 원칙적으로 거절할 수 없다", "판매자가 차량 인도를 제공하면 잔금 지급을 거절할 수 없다"]},
    "WC2-moving-damage": {
        "core_rule": ["채무자가 타인을 사용하여 이행하면 피용자의 고의·과실은 채무자의 고의·과실로 본다(민법 391조)",
                      "업체는 작업자의 과실에 대해 계약상(채무불이행) 책임을 진다"],
        "preparations": ["이사 계약서", "파손 사진", "수리 견적서·영수증"],
        "cautions": ["작업자의 고의나 과실이 있어야 한다", "손해(파손과 손해액)를 증명해야 한다"]},
    "WX1-slap-assault": {
        "core_rule": ["사람의 신체에 폭행을 가한 자는 2년 이하 징역, 500만 원 이하 벌금, 구류 또는 과료에 처한다(형법 260조 1항)",
                      "폭행죄는 피해자의 명시한 의사에 반하여 공소를 제기할 수 없다(260조 3항)"],
        "preparations": ["CCTV·목격자 진술", "사건 경위 기록"],
        "cautions": ["피해자가 처벌을 원하지 않는다는 의사를 밝히면 처벌할 수 없다(반의사불벌)", "상처가 생겼다면 폭행이 아닌 다른 죄가 문제 될 수 있다"]},
    "WX2-mirror-damage": {
        "core_rule": ["타인의 재물을 손괴하여 효용을 해한 자는 3년 이하 징역 또는 700만 원 이하 벌금에 처한다(형법 366조)",
                      "재물손괴죄는 고의가 있어야 성립한다(판결)"],
        "preparations": ["블랙박스·CCTV 영상", "파손 사진", "수리비 견적서"],
        "cautions": ["실수로 부딪쳐 망가뜨린 경우에는 이 죄로 처벌되지 않는다(고의 필요)"]},
    "WA1-disclosure-delay": {
        "core_rule": ["공공기관은 청구를 받은 날부터 10일 이내에 공개 여부를 결정해야 한다(정보공개법 11조 1항)",
                      "부득이하면 10일 범위에서 연장할 수 있으나 연장 사실과 사유를 지체 없이 문서로 통지해야 한다(11조 2항)"],
        "preparations": ["정보공개 청구서와 접수일 기록", "연장 통지를 받았는지 확인할 자료"],
        "cautions": ["연장 통지를 받았다면 결정기간이 최대 10일 늘어난다", "제3자 관련 정보이면 제3자에게 통지할 수 있다(11조 3항)",
                     "다른 기관이 보유한 정보이면 이송되고 그 사실을 통지받는다(11조 4항)"]},
    "WA2-phone-suspension": {
        "core_rule": ["처분은 다른 법령에 특별한 규정이 없으면 문서로 해야 한다(행정절차법 24조 1항)",
                      "긴급하거나 경미한 경우에만 말·전화 등으로 할 수 있고, 요청하면 지체 없이 문서를 주어야 한다(24조 2항)"],
        "preparations": ["통화 기록(일시·내용)", "처분 문서를 요청한 기록"],
        "cautions": ["긴급하거나 경미한 사안이면 문서 없이도 처분할 수 있다", "다른 법령에 특별한 규정이 있으면 예외다",
                     "문서를 요청하면 행정청은 지체 없이 주어야 한다"]},
}


def previous_statutes() -> set[tuple[str, str]]:
    used = set()
    for name in PREVIOUS:
        for s in json.loads((ROOT / "data/eval" / name / "sources.json").read_text()):
            if s["kind"] == "statute":
                used.add((s["law_name"], s["article"]))
    return used


def main() -> None:
    overlap = {(law, number) for _, law, number in STATUTES} & previous_statutes()
    if overlap:
        raise SystemExit(f"statutes already used in earlier sets: {overlap}")
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
        sources.append({"id": cid, "kind": "case", "title": title, "court": "가상지방법원", "case_number": number,
                        "quote": holding, "extraction_file": fname,
                        "sha256": hashlib.sha256(content.encode()).hexdigest()})
    cases = [{"id": cid, "domain": domain, "message": message, "sources": [*statutes, case_id], "case_source": case_id}
             for cid, domain, message, statutes, case_id in CASES]
    assert set(COMPARISON) == {c["id"] for c in cases}
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(COMPARISON, ensure_ascii=False, indent=2) + "\n")
    print(f"built {len(cases)} adapter held-out-2 cases, {len(sources)} sources -> {OUT}")


if __name__ == "__main__":
    main()
