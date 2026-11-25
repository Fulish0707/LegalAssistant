"""Build the held-out set for the scaled consulting adapter decision (data/eval/adapter_consult_heldout3).

Eight new cases; statutes not used in any earlier answer set (checked). Comparison table fixed before any
model run; gates in README.md.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src .venv/bin/python scripts/build_adapter_heldout3.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_answer_eval import FICTIONAL_MARK, article_text  # noqa: E402

OUT = ROOT / "data/eval/adapter_consult_heldout3"
CORPUS = ROOT / "data/api_laws/statute-api-full"
PREVIOUS = ["answer_generation", "answer_prompt_heldout", "answer_prompt_heldout2", "adapter_consult_heldout",
            "adapter_consult_heldout2"]

STATUTES = [
    ("law-628", "민법", "제628조"), ("law-621", "민법", "제621조"),
    ("law-543", "민법", "제543조"), ("law-681", "민법", "제681조"),
    ("law-c329", "형법", "제329조"), ("law-c350", "형법", "제350조"),
    ("law-ap26", "행정절차법", "제26조"), ("law-oi18", "공공기관의 정보공개에 관한 법률", "제18조"),
]

JUDGMENTS = [
    ("case-XL1", "차임감액청구", "2026가단99501",
     "원고는 피고 소유 상가를 월 차임 300만 원에 임차하였다. 주변 공사로 상가 앞 도로가 1년째 통제되어 유동 인구가 크게 줄었고, 같은 건물 다른 점포들의 차임도 30% 정도 하락하였다. 원고는 차임 감액을 청구하였다.",
     "피고는 약정한 차임은 그대로 지급하여야 한다고 주장한다. 원고는 경제사정 변동으로 약정 차임이 상당하지 않게 되었다고 주장한다.",
     "임대물에 대한 공과부담의 증감 기타 경제사정의 변동으로 약정한 차임이 상당하지 아니하게 된 때에는 당사자는 장래에 대한 차임의 증감을 청구할 수 있다(민법 제628조). 도로 통제로 유동 인구가 줄고 인근 점포의 차임이 하락한 것은 경제사정의 변동에 해당하고, 약정 차임 300만 원은 상당하지 않게 되었다. 따라서 원고는 장래에 대하여 차임 감액을 청구할 수 있다. 다만 이미 지급한 차임의 반환을 구하는 것은 아니다.",
     "피고는 장래의 차임을 감액한다."),
    ("case-XL2", "임대차등기협력", "2026가단99502",
     "원고는 피고 소유 건물을 임차하면서 임대차등기에 관하여 아무 약정을 하지 않았다. 원고는 피고에게 임대차등기절차에 협력할 것을 청구하였다.",
     "피고는 등기에 협력할 의무가 없다고 주장한다. 원고는 법률에 따라 협력을 청구할 수 있다고 주장한다.",
     "부동산임차인은 당사자 간에 반대약정이 없으면 임대인에 대하여 그 임대차등기절차에 협력할 것을 청구할 수 있다(민법 제621조 제1항). 부동산임대차를 등기한 때에는 그때부터 제삼자에 대하여 효력이 생긴다(같은 조 제2항). 이 사건 임대차에는 등기에 관한 반대약정이 없으므로 원고는 피고에게 등기절차 협력을 청구할 수 있고, 등기를 마치면 제삼자에게 대항할 수 있다.",
     "피고는 원고의 임대차등기절차에 협력한다."),
    ("case-XC1", "해제의사표시철회", "2026가단99503",
     "원고는 피고의 채무불이행을 이유로 계약 해제 통지를 내용증명으로 보냈고, 통지는 피고에게 도달하였다. 그 후 원고는 마음이 바뀌어 해제를 철회한다고 통지하였다.",
     "원고는 철회하였으므로 계약이 유지된다고 주장한다. 피고는 해제의 의사표시는 철회할 수 없다고 주장한다.",
     "계약 또는 법률의 규정에 의하여 당사자에게 해지 또는 해제의 권리가 있는 때 그 해지 또는 해제는 상대방에 대한 의사표시로 한다(민법 제543조 제1항). 그 의사표시는 철회하지 못한다(같은 조 제2항). 원고의 해제 통지가 피고에게 도달하여 효력이 생겼으므로 원고는 이를 철회할 수 없다. 다만 상대방의 동의 아래 계약을 다시 맺는 것은 별개의 문제이다.",
     "원고의 철회 통지는 효력이 없다."),
    ("case-XC2", "중개사선관의무", "2026가단99504",
     "원고는 피고 공인중개사에게 상가 임차를 의뢰하였다. 피고는 등기사항증명서를 확인하지 않고 중개하였고, 그 건물에는 근저당권이 설정되어 있었다. 이후 경매로 원고는 보증금을 돌려받지 못하였다.",
     "피고는 원고가 직접 확인하였어야 한다고 주장한다. 원고는 중개사가 선량한 관리자의 주의로 확인하였어야 한다고 주장한다.",
     "수임인은 위임의 본지에 따라 선량한 관리자의 주의로써 위임사무를 처리하여야 한다(민법 제681조). 부동산 중개를 의뢰받은 중개사는 권리관계를 확인할 주의의무가 있고, 등기사항증명서의 확인 없이 중개한 것은 그 의무를 다하지 못한 것이다. 따라서 피고는 그 위반으로 원고가 입은 손해를 배상할 책임이 있다. 손해는 중개사의 확인 누락과 보증금 손실 사이의 관련성으로 판단한다.",
     "피고는 원고에게 손해를 배상한다."),
    ("case-XX1", "타인자전거절취", "2026고단99505",
     "피고인은 길가에 잠금 없이 세워 둔 타인의 자전거를 타고 갔다. 자전거는 주인이 잠시 자리를 비운 사이 놓여 있던 것으로, 주인의 관리 아래에 있었다.",
     "피고인은 버려진 것으로 알았다고 주장한다. 검사는 절도죄가 성립한다고 주장한다.",
     "타인의 재물을 절취한 자는 6년 이하의 징역 또는 1천만 원 이하의 벌금에 처한다(형법 제329조). 절도는 타인이 점유하는 재물을 그 의사에 반하여 취득하는 것이다. 자전거는 주인의 관리 아래 있었으므로 타인의 점유를 이탈한 재물이 아니고, 이를 타고 간 것은 절취에 해당한다. 재물이 누구의 관리에도 놓이지 않은 상태였다면 다른 죄가 문제 될 수 있다. 버려진 것으로 알았다는 주장은 고의의 문제이나, 잠금 없이 세워 둔 자전거를 버려진 것으로 보기는 어렵다.",
     "피고인은 절도죄로 유죄이다."),
    ("case-XX2", "협박금원교부", "2026고단99506",
     "피고인은 피해자에게 '말을 듣지 않으면 가족을 해치겠다'고 여러 차례 말하였고, 피해자는 두려운 마음에 500만 원을 건넸다.",
     "피고인은 빌린 돈이라고 주장한다. 검사는 공갈죄가 성립한다고 주장한다.",
     "사람을 공갈하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 10년 이하의 징역 또는 2천만 원 이하의 벌금에 처한다(형법 제350조 제1항). 공갈은 상대방에게 공포심을 일으킬 정도의 해악을 고지하여 재물을 교부받는 것이다. 가족을 해치겠다는 말은 해악의 고지이고, 피해자가 두려운 마음에 돈을 건넸으므로 공갈죄가 성립한다. 해악 고지 없이 돈을 빌린 것이라면 공갈이 아니다. 제삼자에게 재물을 교부하게 한 경우에도 같은 형으로 처벌한다(같은 조 제2항).",
     "피고인은 공갈죄로 유죄이다."),
    ("case-XA1", "불복고지누락", "2026구합99507",
     "피고 구청은 원고에게 과태료 부과 처분을 하면서 처분서에 불복 방법, 청구절차와 청구기간을 전혀 적지 않았다.",
     "피고는 처분의 내용은 적었으므로 문제없다고 주장한다. 원고는 불복 사항을 알려 주지 않았다고 주장한다.",
     "행정청이 처분을 할 때에는 당사자에게 그 처분에 관하여 행정심판 및 행정소송을 제기할 수 있는지 여부, 그 밖에 불복을 할 수 있는지 여부, 청구절차 및 청구기간, 그 밖에 필요한 사항을 알려야 한다(행정절차법 제26조). 피고는 이러한 사항을 전혀 알리지 않았으므로 고지 의무를 위반한 절차적 하자가 있다.",
     "이 사건 처분은 고지 의무 위반의 하자가 있다."),
    ("case-XA2", "정보공개이의신청", "2026구합99508",
     "원고는 피고 구청에 정보공개를 청구하였으나 비공개 결정을 통지받았다. 원고는 통지를 받은 날부터 20일 후에 피고에게 문서로 이의신청을 하였다. 피고는 받은 날부터 10일이 지나도록 결정하지 않았다.",
     "피고는 이의신청을 심사할 필요가 없다고 주장한다. 원고는 법정 기간 내에 결정하여야 한다고 주장한다.",
     "청구인은 비공개 결정 또는 부분 공개 결정에 대하여 불복이 있으면 결정 통지를 받은 날부터 30일 이내에 해당 공공기관에 문서로 이의신청을 할 수 있다(공공기관의 정보공개에 관한 법률 제18조 제1항). 원고의 이의신청은 20일 후에 이루어져 기간 내이고 문서로 하였다. 공공기관은 이의신청을 받은 날부터 7일 이내에 결정하고 그 결과를 지체 없이 문서로 통지하여야 하며, 부득이한 사유가 있으면 7일의 범위에서 연장하고 그 사유를 통지하여야 한다(같은 조 제3항). 피고는 연장 통지 없이 7일을 넘겼다. 이의신청을 각하 또는 기각하면 행정심판이나 행정소송을 제기할 수 있다는 사실을 결과 통지와 함께 알려야 한다(같은 조 제4항).",
     "피고는 이의신청에 대한 법정 결정기간을 지키지 않았다."),
]

CASES = [
    ("XL1-rent-reduction", "lease", "상가를 월 300만 원에 빌렸는데 앞 도로가 1년째 공사 중이라 손님이 뚝 끊겼어요. 같은 건물 다른 가게들은 월세를 30%쯤 깎았대요. 저도 월세를 깎아 달라고 할 수 있나요?", ["law-628"], "case-XL1"),
    ("XL2-lease-registration", "lease", "건물을 빌렸는데 계약서에 임대차등기 이야기는 없었어요. 집주인에게 등기 절차에 협조해 달라고 할 수 있나요?", ["law-621"], "case-XL2"),
    ("XC1-withdraw-rescission", "civil", "상대방이 계약을 이행하지 않아서 해제한다고 내용증명을 보냈고 상대방이 받았어요. 그런데 마음이 바뀌어서 철회한다고 다시 보냈어요. 계약은 다시 살아나나요?", ["law-543"], "case-XC1"),
    ("XC2-broker-duty", "civil", "공인중개사를 통해 상가를 빌렸는데, 중개사가 등기부를 확인하지 않았대요. 그 건물에 근저당이 있어서 경매로 보증금을 못 받았어요. 중개사에게 배상을 받을 수 있나요?", ["law-681"], "case-XC2"),
    ("XX1-bicycle-theft", "criminal", "길가에 잠금 없이 세워 둔 자전거를 타고 갔는데 주인이 잠시 자리를 비운 사이였대요. 절도죄가 되나요, 아니면 유실물 문제인가요?", ["law-c329"], "case-XX1"),
    ("XX2-extortion", "criminal", "어떤 사람이 '말을 듣지 않으면 가족을 해치겠다'고 여러 번 해서 무서운 마음에 500만 원을 줬어요. 이건 어떤 죄가 되나요?", ["law-c350"], "case-XX2"),
    ("XA1-no-appeal-notice", "administrative", "구청에서 과태료 처분서를 받았는데, 이의 제기 방법이나 기간이 아무것도 안 적혀 있어요. 문제없는 건가요?", ["law-ap26"], "case-XA1"),
    ("XA2-disclosure-objection", "administrative", "정보공개 청구가 비공개됐어요. 통지받은 지 20일쯤 뒤에 구청에 이의신청을 문서로 냈는데 열흘이 넘도록 답이 없어요. 제가 늦게 낸 건가요, 구청이 늦는 건가요?", ["law-oi18"], "case-XA2"),
]

COMPARISON = {
    "XL1-rent-reduction": {
        "core_rule": ["공과부담의 증감 기타 경제사정의 변동으로 약정 차임이 상당하지 않게 되면 당사자는 장래에 대한 차임의 증감을 청구할 수 있다(민법 628조)"],
        "preparations": ["임대차계약서(약정 차임)", "인근 점포의 차임 수준을 보여 주는 자료", "도로 통제·공사 기간을 확인할 자료"],
        "cautions": ["감액은 장래에 대한 것이고 이미 지급한 차임의 반환은 아니다", "경제사정 변동으로 약정 차임이 상당하지 않게 되었어야 한다"]},
    "XL2-lease-registration": {
        "core_rule": ["반대약정이 없으면 부동산임차인은 임대인에게 임대차등기절차 협력을 청구할 수 있다(민법 621조 1항)",
                      "임대차를 등기하면 그때부터 제삼자에게 효력이 생긴다(621조 2항)"],
        "preparations": ["임대차계약서(반대약정 유무)"],
        "cautions": ["등기를 하지 않으면 제삼자에게 대항할 수 없다", "반대약정이 있으면 협력을 청구할 수 없다"]},
    "XC1-withdraw-rescission": {
        "core_rule": ["해지·해제는 상대방에 대한 의사표시로 한다(민법 543조 1항)", "그 의사표시는 철회하지 못한다(543조 2항)"],
        "preparations": ["해제 통지 내용증명과 도달 기록", "철회 통지 기록"],
        "cautions": ["도달한 해제 의사표시는 철회할 수 없다", "해제권이 있어야 해제가 유효하다", "상대방 동의 아래 다시 계약하는 것은 별개다"]},
    "XC2-broker-duty": {
        "core_rule": ["수임인은 위임의 본지에 따라 선량한 관리자의 주의로 사무를 처리해야 한다(민법 681조)",
                      "중개사가 권리관계 확인을 누락하면 그로 인한 손해를 배상할 책임이 있다(판결)"],
        "preparations": ["중개 의뢰·계약 기록", "등기사항증명서", "중개사가 권리관계를 설명한 자료(중개대상물 확인·설명서)"],
        "cautions": ["확인 누락과 손해 사이의 관련성이 있어야 한다", "임차인이 이미 알고 있었다면 결론이 달라질 수 있다"]},
    "XX1-bicycle-theft": {
        "core_rule": ["타인의 재물을 절취한 자는 6년 이하 징역 또는 1천만 원 이하 벌금에 처한다(형법 329조)",
                      "타인이 점유하는 재물을 그 의사에 반하여 취득하면 절도이다(판결)"],
        "preparations": ["CCTV·목격자 진술", "자전거 소유·관리 상태를 확인할 자료"],
        "cautions": ["누구의 관리에도 없는 재물이었다면 절도가 아닌 다른 죄가 문제 된다", "버려진 것으로 알았다는 주장은 고의의 문제다"]},
    "XX2-extortion": {
        "core_rule": ["사람을 공갈하여 재물의 교부를 받거나 재산상 이익을 취득한 자는 10년 이하 징역 또는 2천만 원 이하 벌금에 처한다(형법 350조 1항)",
                      "공갈은 공포심을 일으킬 정도의 해악 고지로 재물을 교부받는 것이다(판결)"],
        "preparations": ["협박 발언 기록(문자·녹취)", "송금·교부 기록"],
        "cautions": ["해악 고지가 없으면 공갈이 아니다", "제삼자에게 교부하게 해도 같은 형이다(350조 2항)"]},
    "XA1-no-appeal-notice": {
        "core_rule": ["행정청은 처분 시 행정심판·행정소송 제기 가능 여부, 불복 방법, 청구절차와 청구기간 등을 알려야 한다(행정절차법 26조)"],
        "preparations": ["처분서(불복 안내 기재 부분)"],
        "cautions": ["고지를 빠뜨리면 절차적 하자가 된다", "불복 기간은 별도로 확인해야 한다"]},
    "XA2-disclosure-objection": {
        "core_rule": ["비공개·부분공개 결정에 불복하면 통지받은 날부터 30일 이내에 문서로 이의신청할 수 있다(정보공개법 18조 1항)",
                      "공공기관은 이의신청을 받은 날부터 7일 이내에 결정하고 지체 없이 문서로 통지해야 한다(18조 3항)",
                      "부득이하면 7일 범위에서 연장하고 사유를 통지해야 한다(18조 3항 단서)"],
        "preparations": ["비공개 결정 통지서와 수령일", "이의신청서와 접수일 기록"],
        "cautions": ["이의신청은 30일 이내에 해야 한다", "연장 통지를 받았다면 최대 7일 더 걸릴 수 있다",
                     "각하·기각 시 행정심판·행정소송을 제기할 수 있다(18조 4항)"]},
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
    print(f"built {len(cases)} adapter held-out-3 cases, {len(sources)} sources -> {OUT}")


if __name__ == "__main__":
    main()
