"""Build the consulting-format SFT pilot v2: user-situation questions and consultation answers.

v1 (scripts/build_consult_sft.py) put judgment Q&A answers in the consultation slot; the adapter learned to copy
the evidence judgment's facts and conclusion into the user's case and never to ask follow-ups
(data/eval/adapter_consult_heldout/RESULTS.md). v2 writes new targets from the same real court judgments:

- a local Apache-2.0 model (Qwen3-8B via Ollama) drafts a layperson's question about a situation like the
  judgment's (no party labels, one decisive fact left out) and a consultation answer that keeps the court's
  rule apart from the user's facts, concludes conditionally, asks for the missing fact and lists materials;
- code keeps a draft only if every quote is literally in the passages, no foreign statute/case appears, the
  question and answer carry no party labels, the summary is conditional, follow-up questions exist, materials
  pass the v1 material filter, and the production Verifier passes;
- a second pass of the same model judges fabricated user facts, rule support and conditional conclusion.

Usage:
  python scripts/build_consult_sft_v2.py draft --candidates 600 --out data/training/consult-sft-v2
  python scripts/build_consult_sft_v2.py assemble --out data/training/consult-sft-v2 --train 300 --validation 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_consult_sft import (  # noqa: E402
    Item, Preparation, evidence_bundle, is_material, norm, parse_case, passages, production_messages, sample_rows,
)

SEED = 20261121
DRAFTER = "qwen3:8b"
PARTY = re.compile(r"원고|피고|피고인|소외|청구인|피청구인|이 사건|본건|당심|원심")
CONDITIONAL = re.compile(r"면|경우|라면|다면|수 있|가능성|여부에 따라|달라질")
# Offences struck down or repealed (간통 2015, 혼인빙자간음 2009, 낙태 2021): teaching them as current law is harmful.
ABOLISHED = re.compile(r"간통|혼인빙자|낙태")


def current_law_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not ABOLISHED.search(" ".join((r["title"], r["question"], r["answer"])))]


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=30, max_length=300)
    user_facts: list[str] = Field(min_length=1, max_length=4)
    missing_fact: str = Field(min_length=4, max_length=80)
    legal_issue: str = Field(min_length=4, max_length=80)
    rule: str = Field(min_length=20, max_length=350)
    application: str = Field(min_length=20, max_length=400)
    summary: str = Field(min_length=10, max_length=160)
    preparations: list[Preparation] = Field(default_factory=list, max_length=3)
    cautions: list[Item] = Field(default_factory=list, max_length=3)
    follow_up_questions: list[str] = Field(min_length=1, max_length=3)
    rule_quote: str = Field(min_length=10, max_length=80)


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fabricated_user_fact: bool
    rule_supported: bool
    conclusion_conditional: bool


class StrictJudgement(BaseModel):
    """Second-pass review with reasoning; every field must be true to keep the draft."""
    model_config = ConfigDict(extra="forbid")
    question_coherent: bool
    no_fabricated_user_fact: bool
    rule_matches_source: bool
    application_follows_rule: bool
    conclusion_conditional: bool


DRAFT_SYSTEM = """당신은 판결문을 바탕으로 법률 상담 학습 예제를 만드는 도우미입니다. 주어진 판결 발췌와 참고 답변을 읽고, 비슷한 처지의 일반인이 보낼 상담 질문과 그 상담 답변을 만드세요.

질문(question):
- 판결의 사건과 비슷하지만 구체적 내용(금액·날짜·물건 등)은 바꾼 일반인의 상황을 1인칭 존댓말로 씁니다.
- 원고·피고·피고인·이 사건 같은 재판 용어를 쓰지 마세요.
- 판결의 결론을 바꿀 수 있는 중요한 사실 하나는 일부러 빼고 missing_fact에 적으세요. user_facts에는 질문에 실제로 적은 사실만 적으세요.

답변:
- rule: 판결이 제시한 법리를 "비슷한 사건에서 법원은 ~라고 판단했습니다"처럼 판결의 판단임을 밝혀 씁니다. 판결 사건의 구체적 사실은 쓰지 마세요.
- application: 그 법리를 질문자의 사실(user_facts)에만 적용합니다. 질문에 없는 사실을 질문자의 사실처럼 쓰지 말고, missing_fact에 따라 결론이 어떻게 달라지는지 조건으로 설명하세요. 질문자를 원고·피고라고 부르지 마세요.
- summary: 조건을 붙인 핵심 결론 한두 문장(예: "~라면 ~할 수 있습니다").
- preparations: 질문자가 모을 수 있는 자료. material은 문서·기록·물건의 이름(예: 계약서, 송금 내역, 문자 메시지, 사진, 진단서), proves는 그 자료로 보여 줄 사실, quote는 그 사실이 판단 기준임을 보여 주는 발췌 문장입니다. 해당 없으면 비워 두세요.
- cautions: 결론을 바꿀 수 있는 예외·기한·조건·요건. text는 발췌 내용과 같은 방향이어야 하고, quote는 그 근거 발췌 문장입니다. 해당 없으면 비워 두세요.
- follow_up_questions: missing_fact를 포함해, 결론을 정하려면 질문자에게 더 물어야 할 구체적 질문 1~3개.
- rule_quote: rule의 근거가 되는 발췌 문장.
- legal_issue: 쟁점 한 줄.

규칙:
- 모든 quote는 발췌 문장을 한 글자도 바꾸지 말고 그대로 복사합니다(80자 이내). 당사자의 주장 문장은 quote로 쓰지 마세요.
- 발췌에 없는 조문·사건번호·서류 이름·절차를 만들어 내지 마세요.
- 한국어로, 일반인이 이해할 수 있게 쓰세요."""

JUDGE_SYSTEM = """상담 답변을 점검합니다. 아래 사용자 질문, 판결 발췌의 근거 문장, 답변을 읽고 세 가지를 판정하세요.
- fabricated_user_fact: 답변이 사용자 질문에 없는 사실을 사용자의 사실로 단정하면 true. 판결 사건의 사실을 사용자에게 옮겨 쓴 경우도 true입니다. 조건문("~라면")으로 쓴 가정은 해당하지 않습니다.
- rule_supported: 답변이 설명한 법리가 근거 문장 및 참고 답변과 같은 내용이면 true, 반대이거나 근거에 없는 법리를 단정하면 false.
- conclusion_conditional: 사용자 사건의 결론을 조건 없이 확정하지 않고, 필요한 사실에 따라 달라질 수 있음을 밝히면 true."""


STRICT_SYSTEM = """법률 상담 학습 예제를 엄격하게 검수합니다. 판결 발췌, 판결에 관한 원래 질문과 참고 답변, 그리고 새로 만든 사용자 질문과 상담 답변이 주어집니다.
다음을 각각 판정하세요. 조금이라도 문제가 있으면 false입니다.
- question_coherent: 사용자 질문이 실제 사람이 보낼 법한, 앞뒤가 맞는 하나의 상황과 질문인가. 판결에 관한 원래 질문을 그대로 옮겼거나 상황이 모순되면 false.
- no_fabricated_user_fact: 답변이 사용자 질문에 없는 사실을 사용자의 사실로 단정하지 않는가. 판결 사건의 사실을 사용자에게 옮겨 쓰면 false. "~라면" 가정은 괜찮다.
- rule_matches_source: 답변이 설명한 법리가 판결 발췌와 참고 답변의 법리와 같은가. 다른 법리(예: 식별력 문제를 혼동 문제로 설명)를 섞거나 근거에 없는 조문을 끌어오면 false.
- application_follows_rule: 사용자 사실에 대한 적용이 그 법리에서 논리적으로 따라 나오는가. 판단 기준을 뒤바꾸거나, 결론과 이유가 모순되거나, 질문과 무관한 내용을 덧붙이면 false.
- conclusion_conditional: 결론이 필요한 사실에 따라 달라질 수 있음을 밝히고 조건 없이 확정하지 않는가."""


def strict_listing(draft: Draft, row: dict) -> str:
    return (f"[판결 발췌]\n{row['context'][:3500]}\n[/판결 발췌]\n원래 질문: {row['question']}\n참고 답변: {row['answer']}\n\n"
            f"[새 사용자 질문]\n{draft.question}\n\n[상담 답변]\n{draft.rule}\n{draft.application}\n결론: {draft.summary}\n"
            f"준비물: {[p.item().text for p in draft.preparations]}\n주의점: {[c.text for c in draft.cautions]}\n"
            f"추가 질문: {draft.follow_up_questions}")


def strict_ok(judgement: dict | None) -> bool:
    return bool(judgement) and all(judgement.get(k) for k in StrictJudgement.model_fields)


def has_party_label(text: str) -> bool:
    return bool(PARTY.search(text))


def anchor_quote(quote: str, haystack: str) -> str | None:
    """The literal passage text a quote stands for, or None.

    Drafts often change only the tail (polite ending, the character at the length cap). Accept the longest
    literal match when it covers at least 85% of the quote and 25 characters; paraphrases fall below that.
    """
    from difflib import SequenceMatcher

    q = norm(quote).strip("\"'“”‘’")
    if len(q) < 6 or "주장" in q:
        return None
    if q in haystack:
        return q
    match = SequenceMatcher(None, haystack, q, autojunk=False).find_longest_match(0, len(haystack), 0, len(q))
    if match.size >= max(25, int(len(q) * 0.85)) and match.b == 0:
        return haystack[match.a:match.a + match.size].strip()
    return None


def quote_ok(quote: str, haystack: str) -> bool:
    return anchor_quote(quote, haystack) is not None


def framed_rule(rule: str) -> str:
    """State that the rule comes from the evidence judgment, so it is not read as the user's outcome."""
    rule = rule.strip()
    return rule if re.search(r"법원|판결|판례", rule) else f"관련 판결에 따르면, {rule}"


def foreign_authority(text: str, context: str) -> list[str]:
    from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs

    context_articles = {r.article for r in extract_statute_refs(context)}
    context_cases = set(extract_case_refs(context))
    return ([r.render() for r in extract_statute_refs(text) if r.article not in context_articles]
            + [c for c in extract_case_refs(text) if c not in context_cases])


def check_draft(draft: Draft, row: dict) -> list[str]:
    """Code checks; returns rejection reasons (empty = keep)."""
    haystack = norm(row["context"])
    reasons = []
    if has_party_label(draft.question):
        reasons.append("party_label_in_question")
    for name in ("application", "summary"):
        if has_party_label(getattr(draft, name)):
            reasons.append(f"party_label_in_{name}")
    if not CONDITIONAL.search(draft.summary):
        reasons.append("summary_not_conditional")
    if not quote_ok(draft.rule_quote, haystack):
        reasons.append("rule_quote_not_in_passages")
    text = " ".join([draft.rule, draft.application, draft.summary, *draft.follow_up_questions,
                     *(p.item().text for p in draft.preparations), *(c.text for c in draft.cautions)])
    if foreign_authority(text, row["context"] + " " + row["title"]):
        reasons.append("foreign_authority")
    if not any(q.strip() for q in draft.follow_up_questions):
        reasons.append("no_follow_up")
    return reasons


def kept_items(draft: Draft, row: dict) -> tuple[list[Item], list[Item], list[str]]:
    """Items that pass the literal-quote and material filters; dropped ones are reported, not fatal."""
    haystack = norm(row["context"])
    preps, cauts, dropped = [], [], []
    for p in draft.preparations:
        quote = anchor_quote(p.quote, haystack)
        if quote and is_material(p.material, p.proves) and not has_party_label(p.proves):
            preps.append(Item(text=p.item().text, quote=quote))
        else:
            dropped.append("preparation")
    for c in draft.cautions:
        quote = anchor_quote(c.quote, haystack)
        if quote and not has_party_label(c.text):
            cauts.append(Item(text=c.text.strip(), quote=quote))
        else:
            dropped.append("caution")
    return preps, cauts, dropped


def citations_for(row: dict, quotes: list[str]):
    from legal_rag.schemas.answer import Citation, CitationKind

    court, number = parse_case(row["document_id"])
    reference = f"{court} {number}" if number else f"{row['title']} 판결"
    texts = [norm(p) for p in passages(row["context"])]
    cited: dict[int, str] = {}
    for quote in quotes:
        q = anchor_quote(quote, norm(row["context"]))
        if not q:
            continue
        for index, text in enumerate(texts, 1):
            if q in text and index not in cited:
                cited[index] = q
                break
    return [Citation(source_id=f"S{i}", kind=CitationKind.case, reference=reference, quote=q) for i, q in sorted(cited.items())]


def sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def assemble_target(draft: Draft, row: dict, preps: list[Item], cauts: list[Item]):
    from legal_rag.schemas import LegalAnswer
    from legal_rag.schemas.answer import Confidence

    caution_text = " ".join(sentence(c.text) for c in cauts)
    analysis = framed_rule(draft.rule) + "\n\n" + draft.application.strip()
    if caution_text:
        analysis += "\n\n주의할 점: " + caution_text
    uncertainty = " ".join(filter(None, [f"{draft.missing_fact.strip().rstrip('.')}에 따라 결론이 달라질 수 있습니다.", caution_text]))
    return LegalAnswer(legal_issues=[draft.legal_issue.strip()], analysis=analysis, summary=draft.summary.strip(),
                       possible_actions=[p.text for p in preps],
                       additional_information_needed=[q.strip() for q in draft.follow_up_questions if q.strip()],
                       citations=citations_for(row, [draft.rule_quote] + [i.quote for i in preps + cauts]),
                       uncertainty=uncertainty, confidence=Confidence.medium)


def draft_rows(args) -> None:
    from legal_rag.llm import Message
    from legal_rag.llm.ollama import OllamaChatModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "drafts.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    model = OllamaChatModel(model=args.model, base_url="http://localhost:11434", timeout=900, think=args.think, num_ctx=8192)
    judge = OllamaChatModel(model=DRAFTER, base_url="http://localhost:11434", timeout=600, think=False, num_ctx=8192)
    rows = current_law_rows(sample_rows(args.candidates, exclude_hashes=set()))
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    print(json.dumps({"candidates": len(rows), "already_done": len(done)}), flush=True)
    with path.open("a") as stream:
        for row in rows:
            if row["id"] in done:
                continue
            user = f"[판결 발췌]\n{row['context']}\n[/판결 발췌]\n\n판결에 관한 질문: {row['question']}\n참고 답변: {row['answer']}"
            started, record = time.perf_counter(), {"id": row["id"], "row": row, "drafter": args.model, "think": args.think}
            try:
                response = model.generate([Message(role="system", content=DRAFT_SYSTEM), Message(role="user", content=user)],
                                          response_schema=Draft, temperature=0.3, max_tokens=8192 if args.think else 2048)
                draft = response.parsed if isinstance(response.parsed, Draft) else Draft.model_validate_json(response.text)
                record["draft"] = draft.model_dump()
                record["code_rejections"] = check_draft(draft, row)
                # Judged even when code rejects, so assembly can re-run code checks after filter fixes.
                rule_quote = anchor_quote(draft.rule_quote, norm(row["context"])) or norm(draft.rule_quote)
                listing = (f"사용자 질문: {draft.question}\n근거 문장: {rule_quote}\n참고 답변: {row['answer']}\n\n"
                           f"답변:\n{draft.rule}\n{draft.application}\n결론: {draft.summary}")
                check = judge.generate([Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=listing)],
                                       response_schema=Judgement, temperature=0.0, max_tokens=256)
                record["judgement"] = (check.parsed if isinstance(check.parsed, Judgement)
                                       else Judgement.model_validate_json(check.text)).model_dump()
            except Exception as exc:  # recorded and skipped at assembly
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"id": row["id"], "ok": "error" not in record,
                              "rejections": record.get("code_rejections"), "judgement": record.get("judgement"),
                              "ms": record["elapsed_ms"]}, ensure_ascii=False), flush=True)


def load_review(path: Path | None) -> dict[str, dict] | None:
    """Curator review: {"id": ..., "keep": bool, "reason": ...} per line. Selection only; drafts are never edited."""
    if path is None:
        return None
    return {r["id"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def judged_ok(judgement: dict | None) -> bool:
    return bool(judgement) and not judgement["fabricated_user_fact"] and judgement["rule_supported"] and judgement["conclusion_conditional"]


def assemble(args) -> None:
    from legal_rag.verification import Verifier

    out = Path(args.out)
    records = [json.loads(line) for line in (out / "drafts.jsonl").read_text().splitlines()]
    report = {"records": len(records), "errors": 0, "code_rejected": 0, "code_reasons": {}, "judge_rejected": 0,
              "judge_reasons": {}, "verifier_rejected": 0, "items_dropped": 0, "with_preparations": 0, "with_cautions": 0}
    review = load_review(args.review)
    if review is not None:
        report.update({"review_file": str(args.review), "reviewed": len(review), "review_kept": 0, "review_rejected": 0,
                       "not_reviewed": 0})
    examples = []
    for record in records:
        if "draft" not in record:
            report["errors"] += 1
            continue
        row, draft = record["row"], Draft.model_validate(record["draft"])
        rejections = check_draft(draft, row)
        if rejections:
            report["code_rejected"] += 1
            for reason in rejections:
                report["code_reasons"][reason] = report["code_reasons"].get(reason, 0) + 1
            continue
        if review is not None:
            decision = review.get(record["id"])
            if decision is None:
                report["not_reviewed"] += 1
                continue
            if not decision["keep"]:
                report["review_rejected"] += 1
                continue
            report["review_kept"] += 1
        judgement = record.get("judgement")
        if review is None and not judged_ok(judgement):
            report["judge_rejected"] += 1
            for key, bad in (("fabricated_user_fact", True), ("rule_supported", False), ("conclusion_conditional", False)):
                if judgement and judgement[key] == bad:
                    report["judge_reasons"][key] = report["judge_reasons"].get(key, 0) + 1
            continue
        preps, cauts, dropped = kept_items(draft, row)
        report["items_dropped"] += len(dropped)
        target = assemble_target(draft, row, preps, cauts)
        if not Verifier().verify(target, evidence_bundle(row)).ok:
            report["verifier_rejected"] += 1
            continue
        report["with_preparations"] += bool(preps)
        report["with_cautions"] += bool(cauts)
        messages = production_messages(draft.question, row, target)
        examples.append({"id": row["id"], "domain": row["domain"], "document_id": row["document_id"],
                         "messages": messages + [{"role": "assistant", "content": target.model_dump_json()}],
                         "user_facts": draft.user_facts, "missing_fact": draft.missing_fact,
                         "source_question": row["question"], "reference_answer": row["answer"]})
    total = args.train + args.validation
    if len(examples) < total:
        raise SystemExit(f"only {len(examples)} usable examples for {total}; draft more candidates\n{json.dumps(report, ensure_ascii=False)}")
    rng = random.Random(SEED)
    rng.shuffle(examples)
    splits = {"validation": examples[: args.validation], "train": examples[args.validation: total]}
    hashes = {}
    for name, rows in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out / f"{name}.jsonl").write_text(text)
        hashes[f"{name}.jsonl"] = hashlib.sha256(text.encode()).hexdigest()
    report.update({"usable": len(examples), "train": len(splits["train"]), "validation": len(splits["validation"]),
                   "hashes": hashes, "seed": SEED, "drafter": sorted({r.get("drafter", DRAFTER) for r in records}),
                   "think": sorted({bool(r.get("think")) for r in records}), "human_legal_review": False})
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--candidates", type=int, default=600)
    d.add_argument("--out", required=True)
    d.add_argument("--model", default=DRAFTER)
    d.add_argument("--think", action="store_true")
    d.add_argument("--ids", default="")
    a = sub.add_parser("assemble")
    a.add_argument("--out", required=True)
    a.add_argument("--train", type=int, default=300)
    a.add_argument("--validation", type=int, default=30)
    a.add_argument("--review", type=Path, default=None, help="curator keep/reject jsonl; replaces the model judge")
    args = parser.parse_args()
    {"draft": draft_rows, "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    main()
