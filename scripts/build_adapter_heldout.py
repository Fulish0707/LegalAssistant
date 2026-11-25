"""Build the held-out set for the consulting adapter decision (data/eval/adapter_consult_heldout).

Eight new cases; issues and statutes not used in answer_generation, answer_prompt_heldout or
answer_prompt_heldout2. Comparison table fixed before any model run; gates in README.md.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src .venv/bin/python scripts/build_adapter_heldout.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_answer_eval import FICTIONAL_MARK, article_text  # noqa: E402

OUT = ROOT / "data/eval/adapter_consult_heldout"
CORPUS = ROOT / "data/api_laws/statute-api-full"
PREVIOUS = ["answer_generation", "answer_prompt_heldout", "answer_prompt_heldout2"]

STATUTES = [
    ("law-h6", "주택임대차보호법", "제6조"), ("law-h6-2", "주택임대차보호법", "제6조의2"),
    ("law-s10", "상가건물 임대차보호법", "제10조"),
    ("law-162", "민법", "제162조"), ("law-168", "민법", "제168조"), ("law-178", "민법", "제178조"),
    ("law-758", "민법", "제758조"),
    ("law-c311", "형법", "제311조"), ("law-c312", "형법", "제312조"), ("law-c360", "형법", "제360조"),
    ("law-ap23", "행정절차법", "제23조"), ("law-al18", "행정소송법", "제18조"),
]

JUDGMENTS = [
    ("case-VL1", "묵시적갱신해지", "2026가단99301",
     "원고는 피고 소유 주택을 기간 2년으로 임차하였다. 피고는 기간이 끝나기 6개월 전부터 2개월 전까지 갱신거절이나 조건 변경의 통지를 하지 않았고, 원고도 통지하지 않았다. 기간 만료 후 4개월이 지나 원고는 피고에게 계약 해지를 통지하였다.",
     "피고는 계약이 2년간 갱신되었으므로 원고가 중도에 해지할 수 없다고 주장한다. 원고는 언제든지 해지할 수 있다고 주장한다.",
     "임대인이 임대차기간이 끝나기 6개월 전부터 2개월 전까지 갱신거절 등의 통지를 하지 아니하면 그 기간이 끝난 때에 전 임대차와 동일한 조건으로 다시 임대차한 것으로 보고(주택임대차보호법 제6조 제1항), 이때 존속기간은 2년으로 본다(같은 조 제2항). 그러나 이와 같이 갱신된 경우 임차인은 언제든지 계약해지를 통지할 수 있고, 그 해지는 임대인이 통지를 받은 날부터 3개월이 지나면 효력이 생긴다(제6조의2). 다만 2기의 차임액에 달하도록 연체하는 등 임차인으로서의 의무를 현저히 위반한 임차인에게는 묵시적 갱신이 적용되지 않는다(제6조 제3항). 원고에게 그러한 사정은 없다. 따라서 원고의 해지는 피고가 통지를 받은 날부터 3개월이 지난 때에 효력이 생긴다.",
     "임대차는 피고가 해지 통지를 받은 날부터 3개월이 지난 때에 종료한다."),
    ("case-VL2", "상가갱신요구거절", "2026가합99302",
     "원고는 피고 소유 상가건물을 임차하여 5년째 영업하고 있다. 원고는 임대차기간 만료 3개월 전에 계약갱신을 요구하였다. 피고는 원고가 과거 차임을 2기분 연체한 적이 있다는 이유로 갱신을 거절하였다.",
     "피고는 차임 연체 전력이 있으므로 갱신을 거절할 수 있다고 주장한다. 원고는 연체가 3기에 이르지 않았다고 주장한다.",
     "임대인은 임차인이 임대차기간 만료 6개월 전부터 1개월 전까지 사이에 계약갱신을 요구하면 정당한 사유 없이 거절하지 못한다(상가건물 임대차보호법 제10조 제1항). 임차인이 3기의 차임액에 해당하는 금액에 이르도록 차임을 연체한 사실이 있는 경우 등 같은 항 각 호의 사유가 있으면 거절할 수 있으나, 원고의 연체는 2기분에 그쳐 제1호에 해당하지 않고, 다른 거절 사유도 인정되지 않는다. 갱신요구권은 전체 임대차기간이 10년을 초과하지 않는 범위에서 행사할 수 있는데(같은 조 제2항), 원고의 전체 임대차기간은 이를 넘지 않는다. 따라서 피고의 갱신 거절은 효력이 없고, 임대차는 전 임대차와 동일한 조건으로 갱신된다(같은 조 제3항).",
     "피고의 갱신 거절은 효력이 없고 임대차는 갱신되었다."),
    ("case-VC1", "대여금시효승인", "2026가단99303",
     "원고는 피고에게 1,000만 원을 빌려주었고 변제기로부터 11년이 지났다. 변제기로부터 7년이 지났을 때 피고는 원고에게 100만 원을 갚으면서 나머지도 갚겠다는 문자메시지를 보냈다.",
     "피고는 변제기로부터 10년이 지나 소멸시효가 완성되었다고 주장한다. 원고는 피고의 채무 승인으로 시효가 중단되었다고 주장한다.",
     "채권은 10년간 행사하지 아니하면 소멸시효가 완성한다(민법 제162조 제1항). 소멸시효는 청구, 압류 또는 가압류·가처분, 승인으로 중단된다(제168조). 시효가 중단되면 중단까지 경과한 기간은 산입하지 않고 중단사유가 종료한 때부터 새로 진행한다(제178조 제1항). 피고가 일부를 갚으면서 나머지도 갚겠다고 한 것은 채무의 승인에 해당하므로 시효는 그때 중단되었고, 그때부터 새로 10년이 진행한다. 따라서 이 사건 채권의 소멸시효는 완성되지 않았다.",
     "피고는 원고에게 나머지 900만 원을 지급한다."),
    ("case-VC2", "간판낙하", "2026가단99304",
     "원고는 보도를 걷다가 건물 외벽에 설치된 간판이 떨어져 다쳤다. 건물은 피고1 소유이고, 간판은 건물을 임차한 피고2가 설치하여 관리하였다. 간판 고정 볼트가 부식되어 있었고 피고2는 설치 후 점검을 하지 않았다.",
     "피고2는 건물 소유자인 피고1이 책임져야 한다고 주장한다. 피고1은 간판을 점유·관리한 피고2의 책임이라고 주장한다.",
     "공작물의 설치 또는 보존의 하자로 타인에게 손해를 가한 때에는 공작물점유자가 손해를 배상할 책임이 있고, 점유자가 손해의 방지에 필요한 주의를 해태하지 아니한 때에는 소유자가 배상할 책임이 있다(민법 제758조 제1항). 고정 볼트가 부식된 간판은 보존에 하자가 있고, 간판을 점유·관리한 피고2는 점검을 하지 않아 필요한 주의를 다하지 않았다. 따라서 피고2가 원고의 손해를 배상할 책임이 있고, 피고1은 책임이 없다. 배상한 자는 손해의 원인에 책임 있는 자에게 구상할 수 있다(같은 조 제3항).",
     "피고2는 원고에게 손해를 배상한다. 원고의 피고1에 대한 청구는 기각한다."),
    ("case-VX1", "단체대화방모욕", "2026고단99305",
     "피고인은 20명이 참여한 단체 대화방에서 피해자를 지칭하여 경멸적인 욕설을 게시하였다. 피해자는 사건 후 피고인을 고소하였다.",
     "피고인은 대화방 안의 말에 불과하다고 주장한다. 검사는 모욕죄가 성립한다고 주장한다.",
     "공연히 사람을 모욕한 자는 1년 이하의 징역이나 금고 또는 200만 원 이하의 벌금에 처한다(형법 제311조). 20명이 참여한 대화방에 게시한 욕설은 불특정 또는 다수인이 인식할 수 있는 상태에 놓인 것이어서 공연성이 인정된다. 모욕죄는 고소가 있어야 공소를 제기할 수 있는데(형법 제312조 제1항), 이 사건은 피해자의 고소가 있었다. 따라서 모욕죄가 성립한다.",
     "피고인은 모욕죄로 유죄이다."),
    ("case-VX2", "습득지갑", "2026고단99306",
     "피고인은 도로에 떨어져 있던 타인의 지갑을 주워 그 안의 현금 30만 원을 사용하였다. 지갑은 누구의 관리에도 놓여 있지 않은 상태였다. 피고인은 수사 중 피해자에게 30만 원을 반환하였다.",
     "피고인은 주운 돈이라 죄가 되지 않는다고 주장한다. 검사는 점유이탈물횡령죄가 성립한다고 주장한다.",
     "유실물 또는 타인의 점유를 이탈한 재물을 횡령한 자는 1년 이하의 징역이나 300만 원 이하의 벌금 또는 과료에 처한다(형법 제360조 제1항). 도로에 떨어져 누구의 관리에도 놓여 있지 않던 지갑은 타인의 점유를 이탈한 재물이고, 피고인이 그 안의 현금을 사용한 것은 횡령에 해당한다. 다만 지갑이 다른 사람의 관리 아래 있었다면 점유이탈물이 아니어서 이 죄가 아닌 다른 죄가 문제 될 수 있다. 양형에서는 피고인이 피해자에게 전액을 반환한 점을 참작한다.",
     "피고인은 점유이탈물횡령죄로 유죄이다."),
    ("case-VA1", "처분이유미제시", "2026구합99307",
     "피고 구청은 원고의 영업허가를 취소하면서 처분서에 '관계 법령 위반'이라고만 적었다. 원고는 허가 취소를 신청한 적이 없고, 이 처분은 단순·반복적이거나 경미한 처분이 아니었으며 긴급히 처분할 필요도 없었다.",
     "원고는 처분의 근거와 이유가 제시되지 않아 위법하다고 주장한다. 피고는 원고가 위반 사실을 알고 있었다고 주장한다.",
     "행정청은 처분을 할 때에는 당사자에게 그 근거와 이유를 제시하여야 한다(행정절차법 제23조 제1항). 신청 내용을 모두 그대로 인정하는 처분, 단순·반복적인 처분 또는 경미한 처분으로서 당사자가 그 이유를 명백히 알 수 있는 경우, 긴급히 처분을 할 필요가 있는 경우는 예외이나(같은 항 각 호), 이 사건은 어느 경우에도 해당하지 않는다. 처분서에 '관계 법령 위반'이라고만 적은 것으로는 어떤 근거와 이유로 처분하였는지 알 수 없다. 따라서 이 사건 처분은 이유 제시 의무를 위반한 절차적 하자가 있다.",
     "이 사건 처분은 위법하다."),
    ("case-VA2", "행정심판임의전치", "2026구합99308",
     "원고의 건축허가 신청에 대하여 피고가 거부 처분을 하였다. 원고는 행정심판을 거치지 않고 곧바로 취소소송을 제기하였다. 처분의 근거 법률에는 행정심판의 재결을 거쳐야 취소소송을 제기할 수 있다는 규정이 없다.",
     "피고는 행정심판을 먼저 거쳐야 한다고 주장한다. 원고는 바로 소송을 낼 수 있다고 주장한다.",
     "취소소송은 법령에 따라 행정심판을 제기할 수 있는 경우에도 이를 거치지 아니하고 제기할 수 있다(행정소송법 제18조 제1항 본문). 다만 다른 법률에 행정심판의 재결을 거치지 아니하면 취소소송을 제기할 수 없다는 규정이 있으면 그러하지 아니하고(같은 항 단서), 이 경우에도 같은 조 제2항·제3항의 사유가 있으면 재결 없이 또는 심판 없이 소를 제기할 수 있으며 그 사유는 소명하여야 한다(같은 조 제4항). 이 사건 처분의 근거 법률에는 그러한 규정이 없으므로 원고는 행정심판을 거치지 않고 취소소송을 제기할 수 있다.",
     "원고의 소는 적법하다."),
]

CASES = [
    ("VL1-tacit-renewal-exit", "lease", "2년 전세 계약이 끝났는데 집주인이 만료 전에 아무 연락이 없었고 저도 아무 말 안 했어요. 만료 후 4개월이 지났는데 이제 이사를 가고 싶어요. 바로 나갈 수 있나요?", ["law-h6", "law-h6-2"], "case-VL1"),
    ("VL2-shop-renewal-refusal", "lease", "상가를 빌려서 5년째 장사하고 있어요. 계약 만료 3개월 전에 갱신해 달라고 했더니, 예전에 월세를 두 번 밀린 적이 있다며 주인이 거절해요. 거절할 수 있는 건가요?", ["law-s10"], "case-VL2"),
    ("VC1-loan-limitation", "civil", "친구에게 1,000만 원을 빌려줬는데 갚기로 한 날로부터 11년이 지났어요. 그런데 4년 전에 친구가 100만 원을 갚으면서 나머지도 꼭 갚겠다고 문자를 보냈어요. 아직 받을 수 있나요?", ["law-162", "law-168", "law-178"], "case-VC1"),
    ("VC2-falling-sign", "civil", "길을 걷다가 건물 외벽 간판이 떨어져서 다쳤어요. 건물은 다른 사람 소유이고 간판은 1층 가게가 단 거래요. 누구에게 배상을 받아야 하나요?", ["law-758"], "case-VC2"),
    ("VX1-groupchat-insult", "criminal", "20명 있는 단체 대화방에서 누가 저를 두고 심한 욕을 했어요. 처벌받게 할 수 있나요?", ["law-c311", "law-c312"], "case-VX1"),
    ("VX2-found-wallet", "criminal", "길에 떨어진 지갑을 주워서 안에 있던 현금 30만 원을 써 버렸어요. 어떤 처벌을 받게 되나요?", ["law-c360"], "case-VX2"),
    ("VA1-no-reasons", "administrative", "구청이 제 영업허가를 취소했는데 처분서에 '관계 법령 위반'이라고만 적혀 있어요. 무슨 위반인지도 모르겠어요. 다툴 수 있나요?", ["law-ap23"], "case-VA1"),
    ("VA2-direct-suit", "administrative", "건축허가 신청이 거부됐어요. 소송을 하려면 꼭 행정심판부터 거쳐야 하나요?", ["law-al18"], "case-VA2"),
]

COMPARISON = {
    "VL1-tacit-renewal-exit": {
        "core_rule": ["임대인이 기간 만료 6개월 전부터 2개월 전까지 갱신거절 등을 통지하지 않으면 전 임대차와 같은 조건으로 갱신된 것으로 본다(주택임대차보호법 6조 1항)",
                      "묵시적으로 갱신된 경우 임차인은 언제든지 해지를 통지할 수 있다(6조의2 1항)",
                      "해지는 임대인이 통지를 받은 날부터 3개월이 지나면 효력이 생긴다(6조의2 2항)"],
        "preparations": ["임대차계약서(기간)", "임대인의 갱신거절 통지가 없었음을 보여 주는 기록(문자·통화 기록)", "해지 통지를 보낸 날짜를 확인할 수 있는 기록"],
        "cautions": ["해지 효력은 통지 후 바로가 아니라 임대인이 받은 날부터 3개월 뒤에 생긴다",
                     "2기 차임 연체 등 의무를 현저히 위반한 임차인에게는 묵시적 갱신이 적용되지 않는다(6조 3항)"]},
    "VL2-shop-renewal-refusal": {
        "core_rule": ["임차인이 만료 6개월 전부터 1개월 전까지 갱신을 요구하면 임대인은 정당한 사유 없이 거절하지 못한다(상가건물 임대차보호법 10조 1항)",
                      "3기의 차임액에 이르도록 연체한 사실이 있는 경우 등 각 호의 사유가 있으면 거절할 수 있다(10조 1항 단서)",
                      "갱신요구권은 전체 임대차기간 10년을 넘지 않는 범위에서 행사할 수 있다(10조 2항)"],
        "preparations": ["임대차계약서(최초 계약일·기간)", "차임 지급 내역(연체 기간·금액)", "갱신을 요구한 시점과 내용의 기록"],
        "cautions": ["연체가 3기에 이르렀다면 거절할 수 있다", "갱신 요구는 만료 6개월 전부터 1개월 전까지 해야 한다",
                     "전체 임대차기간이 10년을 넘으면 갱신요구권이 없다", "무단 전대·고의 파손 등 다른 거절 사유가 있으면 거절할 수 있다"]},
    "VC1-loan-limitation": {
        "core_rule": ["채권은 10년간 행사하지 않으면 소멸시효가 완성한다(민법 162조 1항)",
                      "소멸시효는 청구, 압류·가압류·가처분, 승인으로 중단된다(168조)",
                      "중단되면 그때까지의 기간은 산입하지 않고 중단사유가 끝난 때부터 새로 진행한다(178조 1항)"],
        "preparations": ["차용증·송금 기록(대여 사실과 변제기)", "일부 변제 기록(100만 원 입금 내역)", "나머지를 갚겠다는 문자메시지"],
        "cautions": ["시효는 승인한 때부터 새로 10년이 진행하므로 그 안에 권리를 행사해야 한다",
                     "승인 등 중단 사유가 없었다면 10년 경과로 시효가 완성된다"]},
    "VC2-falling-sign": {
        "core_rule": ["공작물의 설치·보존의 하자로 손해를 가하면 점유자가 배상 책임을 진다(민법 758조 1항)",
                      "점유자가 손해 방지에 필요한 주의를 다했으면 소유자가 배상 책임을 진다(758조 1항 단서)",
                      "배상한 자는 손해의 원인에 책임 있는 자에게 구상할 수 있다(758조 3항)"],
        "preparations": ["사고 현장·간판 상태 사진", "진단서·치료비 영수증", "간판의 설치·관리 주체를 확인할 자료"],
        "cautions": ["간판의 설치·보존에 하자가 있어야 한다", "점유자가 필요한 주의를 다했다면 소유자에게 청구해야 한다"]},
    "VX1-groupchat-insult": {
        "core_rule": ["공연히 사람을 모욕한 자는 1년 이하 징역이나 금고 또는 200만 원 이하 벌금에 처한다(형법 311조)",
                      "모욕죄는 고소가 있어야 공소를 제기할 수 있다(312조 1항)",
                      "다수가 인식할 수 있는 대화방 게시는 공연성이 인정될 수 있다"],
        "preparations": ["대화방 캡처(발언 내용·일시·작성자)", "대화방 참여 인원을 확인할 자료"],
        "cautions": ["피해자의 고소가 있어야 처벌할 수 있다(친고죄)", "공연성(다수가 인식할 수 있는 상태)이 인정되어야 한다"]},
    "VX2-found-wallet": {
        "core_rule": ["유실물·점유이탈물을 횡령한 자는 1년 이하 징역이나 300만 원 이하 벌금 또는 과료에 처한다(형법 360조 1항)",
                      "누구의 관리에도 놓이지 않은 떨어진 지갑은 점유이탈물이다"],
        "preparations": ["습득 장소·경위를 확인할 자료", "피해자에게 반환한 기록"],
        "cautions": ["지갑이 다른 사람의 관리 아래 있었다면 이 죄가 아닌 다른 죄가 문제 될 수 있다", "반환 사실은 양형에서 참작될 수 있다"]},
    "VA1-no-reasons": {
        "core_rule": ["행정청은 처분할 때 당사자에게 근거와 이유를 제시해야 한다(행정절차법 23조 1항)",
                      "신청을 그대로 인정하는 처분, 단순·반복·경미하고 이유가 명백한 처분, 긴급한 처분은 예외다(23조 1항 각 호)",
                      "'관계 법령 위반'이라고만 적은 것으로는 이유 제시가 되지 않는다(판결)"],
        "preparations": ["처분서(기재된 근거·이유 부분)", "처분 전후 행정청과 주고받은 통지·서면"],
        "cautions": ["예외 사유(단순·반복·경미, 긴급 등)에 해당하면 이유 제시 없이 처분할 수 있다",
                     "단순·반복·경미 또는 긴급 사유로 이유를 생략했더라도 처분 후 요청하면 근거와 이유를 제시해야 한다(23조 2항)"]},
    "VA2-direct-suit": {
        "core_rule": ["취소소송은 행정심판을 거치지 않고 제기할 수 있다(행정소송법 18조 1항 본문)",
                      "다른 법률에 재결을 거쳐야 한다는 규정이 있으면 예외다(18조 1항 단서)"],
        "preparations": ["거부 처분서(처분의 근거 법률 기재)"],
        "cautions": ["처분의 근거 법률에 행정심판 전치 규정이 있으면 심판을 먼저 거쳐야 한다",
                     "전치 규정이 있어도 18조 2항·3항의 사유가 있으면 예외이며 그 사유를 소명해야 한다"]},
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
    print(f"built {len(cases)} adapter held-out cases, {len(sources)} sources -> {OUT}")


if __name__ == "__main__":
    main()
