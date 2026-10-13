"""Flag statute articles and case numbers an answer cites that no evidence contains.

The prompt forbids citing what retrieval did not return, but qwen3:4b still invented
"민법 제543조, 제544조" when the legal search was down (2026-09-27). This check does not
depend on the model: anything cited that the turn's evidence, the user's own text, or
the draft does not contain gets a visible "확인 필요" notice. Mirrors the patterns in
legal_rag.textutil.refs, which this runtime cannot import.
"""

import re

_LAW = r"(?P<law>[가-힣A-Za-z0-9·]+?(?:법|령|규칙|조례|규정))"
_ARTICLE = r"(?P<art>제\s*\d+\s*조(?:\s*의\s*\d+)?)"
STATUTE_RE = re.compile(rf"(?:{_LAW}\s*)?{_ARTICLE}")
_CASE_TYPES = (
    "헌가|헌나|헌다|헌라|헌마|헌바|헌사|헌아|가합|가단|가소|고합|고단|고정|다카|"
    "다|도|두|누|나|라|마|노|오|모|르|므|드|느|브|스|수|무|어|우|주|구|카|타|허|후|헌|초|즈|재|하|바"
)
CASE_RE = re.compile(rf"(?<![0-9])\d{{2,4}}(?:{_CASE_TYPES})\d{{1,7}}(?![0-9])")


def _squash(text):
    return re.sub(r"\s+", "", text or "")


# Korean statutes are cited both by full title and by the common short form:
# 여성기업지원에 관한 법률 -> 여성기업지원법. Comparing both sides after this rewrite lets
# either spelling match the other; the article number must still match separately.
_TITLE_TAILS = ((r"등?에관한특별법", "특별법"), (r"등?에관한법률", "법"), (r"등?에관한법", "법"), (r"법률", "법"))


def _short_title(squashed):
    for pattern, replacement in _TITLE_TAILS:
        squashed = re.sub(pattern, replacement, squashed)
    return squashed


# Words the law-name pattern matches that refer to a law without naming it.
_GENERIC_LAW_WORDS = {"법", "법령", "법률", "동법", "본법", "이법", "같은법", "관련법", "관련법령", "시행령", "시행규칙", "규정", "규칙",
                      "구법", "신법", "현행법", "개정법", "종전법"}


def unsupported_citations(answer, evidence_texts, own_clauses=()):
    """Citations in answer (in order, deduplicated) that no evidence text contains.

    `own_clauses` are texts of the user's own documents. Their clause numbers vouch only for
    an article the answer names without a law.
    """
    own = _squash(" ".join(own_clauses))
    evidence = _squash(" ".join(t for t in evidence_texts if t))
    evidence_short = _short_title(evidence)
    missing = []
    for match in STATUTE_RE.finditer(answer or ""):
        article, law = _squash(match.group("art")), match.group("law")
        if law and _squash(law) in _GENERIC_LAW_WORDS:
            law = None
        # 제3조 must not be satisfied by 제3조의3, but "제124조의 규정" is 제124조.
        law_found = not law or _squash(law) in evidence or _short_title(_squash(law)) in evidence_short
        found = re.search(re.escape(article) + r"(?!의\d)", evidence if law else evidence + own) and law_found
        label = f"{law} {match.group('art')}" if law else match.group("art")
        if not found and label not in missing:
            missing.append(label)
    for case in CASE_RE.findall(answer or ""):
        if case not in evidence and case not in missing:
            missing.append(case)
    return missing


def named_citations(text):
    """Articles named with their law, and case numbers, in text.

    A bare 제4조 is left out: a contract numbers its own clauses that way, and that must not
    vouch for some law's 제4조.
    """
    named = [match.group(0) for match in STATUTE_RE.finditer(text or "")
             if match.group("law") and _squash(match.group("law")) not in _GENERIC_LAW_WORDS]
    return named + CASE_RE.findall(text or "")


def annotate_unsupported(answer, missing):
    if not missing:
        return answer
    return (
        f"{answer}\n\n※ 확인 필요: {', '.join(missing)}은(는) 이번 답변에 쓰인 검색 근거에서 "
        "확인되지 않았습니다. 국가법령정보센터(law.go.kr)나 대법원 종합법률정보에서 직접 확인하세요."
    )


# A time span, a date, an amount or a rate. The model sometimes misuses a figure that is in its
# sources: the user's "두 달 전" became a "2개월 이내" deadline, and a lease's two dates became
# one wrong period. Comparing text cannot catch that (memory.md, 2026-10-06), so the reader is
# told to check, and only when the answer has such a figure.
_FIGURE_RE = re.compile(r"\d[\d,]*\s*(?:억|천만|백만|만)?\s*(?:시간|일|주|개월|달|년|원|%|퍼센트)|(?:한|두|세|석|네|넉|다섯|여섯)\s*(?:달|주|해)")
FIGURE_NOTICE = "※ 기간·날짜·금액은 잘못 옮겨질 수 있습니다. 중요한 결정 전에는 근거 원문과 올리신 자료에서 직접 확인하세요."


def annotate_figures(answer):
    return f"{answer}\n\n{FIGURE_NOTICE}" if _FIGURE_RE.search(answer or "") else answer
