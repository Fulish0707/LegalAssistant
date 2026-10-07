"""Conservative evidence roles derived from explicit section headings only."""
from __future__ import annotations

import re

ROLE_LABELS = {
    "party_claim": "당사자 주장 (채택 여부 미확인)",
    "finding": "기관·법원 판단",
    "legal_rule": "법령·관련 법리",
    "facts": "사실관계",
    "order": "주문·결론",
    "mixed": "복수 역할 (본문 구분 확인 필요)",
    "unknown": "역할 미상 (판단·주장 구분 미확인)",
}


def section_role(headings: list[str], *, statute: bool = False) -> str:
    if statute:
        return "legal_rule"
    role = "unknown"
    for heading in headings:
        title = re.sub(r"\s+", "", heading)
        if "–" in title or "·" in title:
            # A range can include a claim and a decision: never assign it one speaker.
            if any(word in title for word in ("주장", "판단", "주문", "청구취지")):
                role = "mixed"
                continue
        if any(word in title for word in ("판단", "법원의결정", "위원회의결정")):
            role = "finding"
        elif any(word in title for word in ("관련법리", "관계법령", "법령의적용")):
            role = "legal_rule"
        elif any(word in title for word in ("주장", "항변", "진정요지", "청구취지", "항소취지", "상고취지")):
            role = "party_claim"
        elif any(word in title for word in ("인정사실", "기초사실", "사실관계", "사건개요")):
            role = "facts"
        elif title.strip("[]【】()") in ("주문", "결론") or re.search(r"\d+\.(주문|결론)$", title):
            role = "order"
    return role
