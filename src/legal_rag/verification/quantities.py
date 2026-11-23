"""Quantity relation checks: does the answer restate a period, count or amount with the relation its source gave?

Measured 2026-10-07: a plain "is this number in a source" check caught 7 of 384 quantities (3 false
alarms) and only 1 of 4 seen error kinds, because the model rarely invents numbers. It moves a number
that *is* in the source into a different relation: "2기의 차임액에 달하는 때" becomes "2기를 초과해야",
"그 기간을 2년으로 본다" becomes "만료 후 추가로 2년", the user's "두 달 전" becomes "2개월 이내".
These checks compare the relation words around the same quantity in the answer and in its sources.

Rules (each finding names one):
- R1: a quantity of the evidence restated with a conflicting relation (reach vs exceed, additional).
- R2: a point in time of the user ("두 달 전") restated as a window ("2개월 이내").
- R3: a quantity in no source that cannot be computed from two source quantities.

Measurement only until the preregistered gates in data/eval/numeric_check/PROTOCOL.md pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

_SPELLED = {"한": 1, "두": 2, "세": 3, "석": 3, "네": 4, "넉": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_UNIT_ALIASES = {"달": "개월", "해": "년", "주일": "주", "퍼센트": "%", "번": "회", "차례": "회"}
_SCALES = {"억": 100_000_000, "천만": 10_000_000, "백만": 1_000_000, "만": 10_000, "천": 1_000}
# (dimension, factor to the dimension's base unit)
_DIMENSIONS = {"개월": ("month", 1), "년": ("month", 12), "일": ("day", 1), "주": ("day", 7), "시간": ("hour", 1),
               "기": ("count", 1), "회": ("count", 1), "원": ("won", 1), "%": ("percent", 1)}
# "연 5푼", "1할", "20분의 1" are rates in statutes and judgments.
_RATE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(푼|할|리)(?=$|[^가-힣]|[이의으을를에은는과와도로])")
_RATE_FACTORS = {"푼": 1.0, "할": 10.0, "리": 0.1}
_FRACTION = re.compile(r"(?<![\d.])(\d+)\s*분의\s*(\d+)")
# A number the answer assumes for an example ("50만 원이라면", "60만 원인 경우") is a premise, not a claim.
_PREMISE_AFTER = re.compile(r"^\s*(?:이?라면|(?:이었|였)다면|인\s*경우|일\s*경우|일\s*때)")
_EXAMPLE = re.compile(r"예를\s*들어|예컨대|가령|예\s*:|\(예")

_DATE = re.compile(r"(\d{4})\s*(?:년|\.|-)\s*(\d{1,2})\s*(?:월|\.|-)\s*(?:(\d{1,2})\s*일?)?")
_NUMBER = re.compile(r"(?<![\d.제])(\d[\d,]*(?:\.\d+)?)\s*(억|천만|백만|만|천)?\s*(시간|주일|개월|퍼센트|차례|달|일|주|년|해|기|회|번|원|%)(?![가-힣]*조\b)")
_WORDS = re.compile(rf"(?<![가-힣])({'|'.join(sorted(_SPELLED, key=len, reverse=True))})\s*(달|개월|주일|주|해|시간|번|차례|기)(?![가-힣]{{0,1}}째)")
_ARTICLE_AFTER = re.compile(r"^\s*(?:조|항|호)")

EXCEED, WITHIN, REACH, LESS, AGO, ELAPSE, DEEMED, ADD = "exceed", "within", "reach", "less", "ago", "elapse", "deemed", "add"
_AFTER_PATTERNS = [
    (EXCEED, re.compile(r"초과|넘(?!지\s*않)")),
    (WITHIN, re.compile(r"이내|내에|내로|안에|이하|넘지\s*않|까지")),
    (REACH, re.compile(r"이상|달하|달한|달할|이르|이른|에\s*달")),
    (LESS, re.compile(r"미만")),
    (DEEMED, re.compile(r"^\s*(?:으로|로)\s*(?:본다|봅니다|보며|보고|보아|간주)")),
    (AGO, re.compile(r"^\s*(?:쯤|정도|가량)?\s*전(?:에|부터|쯤|이|입|인|으로|[,.\s]|$)")),
    (ELAPSE, re.compile(r"^\s*(?:이|가|을|를)?\s*(?:지나|지난|경과)|^\s*(?:쯤|정도)?\s*(?:후|뒤)")),
    (ADD, re.compile(r"^\s*(?:을|를)?\s*더(?:\s|$)|^\s*추가")),
]
_ADD_BEFORE = re.compile(r"(?:추가로|추가|더)\s*$")
_IGNORED = {("percent", 100.0), ("count", 1.0)}  # "100% 확정할 수 없다", "1회", "한 번"


@dataclass(frozen=True)
class Quantity:
    value: str  # normalized: "2개월", "300만원", "2024-09-01"
    key: tuple[str, float]  # (dimension, amount in base unit) for comparison
    start: int
    end: int
    relations: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Finding:
    rule: str
    quantity: str
    sentence: str
    detail: str


def _relations(text: str, start: int, end: int) -> frozenset[str]:
    after = re.split(r"[.!?\n]|\d", text[end:end + 16], maxsplit=1)[0]
    found = {name for name, pattern in _AFTER_PATTERNS if pattern.search(after)}
    if _ADD_BEFORE.search(text[max(0, start - 8):start]):
        found.add(ADD)
    return frozenset(found)


def extract_quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    taken: list[tuple[int, int]] = []
    for m in _DATE.finditer(text):
        year, month, day = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= month <= 12:
            continue
        value = f"{year}-{month:02d}" + (f"-{int(day):02d}" if day else "")
        out.append(Quantity(value, ("date", float(year + f"{month:02d}" + (f"{int(day):02d}" if day else "00"))),
                            m.start(), m.end()))
        taken.append(m.span())

    def free(span: tuple[int, int]) -> bool:
        return not any(a < span[1] and span[0] < b for a, b in taken)

    for m in _FRACTION.finditer(text):
        denominator, numerator = int(m.group(1)), int(m.group(2))
        if denominator:
            percent = round(numerator / denominator * 100, 6)
            out.append(Quantity(f"{percent:g}%", ("percent", percent), m.start(), m.end(), _relations(text, m.start(), m.end())))
            taken.append(m.span())
    for m in _RATE.finditer(text):
        if free(m.span()):
            percent = round(float(m.group(1)) * _RATE_FACTORS[m.group(2)], 6)
            out.append(Quantity(f"{percent:g}%", ("percent", percent), m.start(), m.end(), _relations(text, m.start(), m.end())))
            taken.append(m.span())
    for m in _NUMBER.finditer(text):
        if not free(m.span()) or _ARTICLE_AFTER.match(text[m.end():]):
            continue
        number, scale, unit = m.group(1).replace(",", ""), m.group(2) or "", _UNIT_ALIASES.get(m.group(3), m.group(3))
        amount = float(number) * (_SCALES.get(scale, 1))
        dimension, factor = _DIMENSIONS[unit]
        value = f"{number.rstrip('0').rstrip('.') if '.' in number else number}{scale}{unit}"
        out.append(Quantity(value, (dimension, amount * factor), m.start(), m.end(), _relations(text, m.start(), m.end())))
        taken.append(m.span())
    for m in _WORDS.finditer(text):
        # "한 달 차임", "한 번" name a unit or an occurrence, not a claimed quantity.
        if not free(m.span()) or m.group(1) == "한":
            continue
        amount, unit = _SPELLED[m.group(1)], _UNIT_ALIASES.get(m.group(2), m.group(2))
        dimension, factor = _DIMENSIONS[unit]
        out.append(Quantity(f"{amount}{unit}", (dimension, float(amount * factor)), m.start(), m.end(),
                            _relations(text, m.start(), m.end())))
        taken.append(m.span())
    return sorted(out, key=lambda q: q.start)


def _sentence(text: str, quantity: Quantity) -> str:
    left = max(text.rfind(c, 0, quantity.start) for c in ".!?\n")
    rights = [i for i in (text.find(c, quantity.end) for c in ".!?\n") if i != -1]
    return text[left + 1:(min(rights) + 1 if rights else len(text))].strip()


def _relations_by_key(quantities: list[Quantity]) -> dict[tuple[str, float], set[str]]:
    by_key: dict[tuple[str, float], set[str]] = {}
    for q in quantities:
        by_key.setdefault(q.key, set()).update(q.relations)
    return by_key


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 0.01 * abs(b), 0.0005)


def _derivable(key: tuple[str, float], sources: set[tuple[str, float]]) -> bool:
    dimension, amount = key
    pool = sorted({a for d, a in sources if d == dimension} | ({100.0} if dimension == "percent" else set()))
    if any(abs(a + b - amount) < 1e-9 or abs(abs(a - b) - amount) < 1e-9 for a, b in combinations(pool, 2)):
        return True
    if any(abs(2 * a - amount) < 1e-9 for a in pool):
        return True
    # an annual rate restated per month, or the reverse ("연 5%" -> "월 약 0.417%")
    return dimension == "percent" and any(_close(amount, a / 12) or _close(amount, a * 12) for a in pool)


def check_quantities(answer: str, evidence: list[str], user: str) -> list[Finding]:
    """Findings for quantities the answer restates with a relation its sources do not support."""
    evidence_quantities = [q for text in evidence for q in extract_quantities(text)]
    user_quantities = extract_quantities(user)
    answer_quantities = extract_quantities(answer)
    premises = {q.key for q in answer_quantities if _PREMISE_AFTER.match(answer[q.end:])}
    by_evidence, by_user = _relations_by_key(evidence_quantities), _relations_by_key(user_quantities)
    every_source = set(by_evidence) | set(by_user)
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    def add(rule: str, q: Quantity, detail: str) -> None:
        if (rule, q.value) not in seen:
            seen.add((rule, q.value))
            findings.append(Finding(rule, q.value, _sentence(answer, q), detail))

    for q in answer_quantities:
        if q.key in _IGNORED:
            continue
        if q.key in by_evidence:
            ev = by_evidence[q.key]
            if EXCEED in q.relations and REACH in ev and EXCEED not in ev:
                add("R1", q, "근거는 '달함·이상', 답변은 '초과'")
            if REACH in q.relations and EXCEED in ev and REACH not in ev:
                add("R1", q, "근거는 '초과', 답변은 '달함·이상'")
            # "그 기간을 2년으로 본다" is a total; "추가로 2년" adds it on top.
            if ADD in q.relations and DEEMED in ev and ADD not in ev:
                add("R1", q, "근거는 총 기간('으로 본다'), 답변은 '추가' 기간")
        if q.key in by_user and WITHIN in q.relations:
            us = by_user[q.key]
            if (AGO in us or ELAPSE in us) and WITHIN not in us and WITHIN not in by_evidence.get(q.key, set()):
                add("R2", q, "사용자는 시점('전'·'지난'), 답변은 기간 창('이내')")
        if q.key in every_source or q.key in premises:
            continue
        if q.key[0] == "date":
            if not _EXAMPLE.search(_sentence(answer, q)):
                add("R3", q, "근거·사용자 글에 없는 날짜")
        elif not _derivable(q.key, every_source | premises):
            add("R3", q, "근거·사용자 글에 없는 수량")
    return findings
