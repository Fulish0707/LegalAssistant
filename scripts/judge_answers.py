#!/usr/bin/env python
"""Grade saved /v1/chat answers against the AIHub reference answers.

`pairs` joins a verify.jsonl from eval_chat_pipeline.py with the reference answers in the
upstream qa.jsonl files (the eval sets leave them out) and marks whether the question's gold
document was among the turn's evidence. The references are written from that one document, so
an answer built on other evidence can be sound and still miss them; report the two groups apart.
`judge` defaults to a versioned rubric separating conclusion, legal basis and evidence
support. `report` prints these metrics separately and does not count failed judge calls
as wrong answers. --rubric legacy preserves the historical two-field scoring.

This measures agreement with the reference answers, not legal correctness.

Example (one heavy process at a time; unload other Ollama models first):
    uv run python scripts/judge_answers.py pairs --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles
    uv run python scripts/judge_answers.py judge --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles
    uv run python scripts/judge_answers.py report --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles \
        --manual data/eval/reports/chat_pipeline/formal-2026-09-28-articles/manual_grades.jsonl

Use --rubric legacy for the original answer_judgments.jsonl reports.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
QA_ROOT = Path.home() / "Documents/legal-rag/data/processed"
SOURCES = [(f"{d}-pilot-200", ROOT / f"data/eval/{d}-pilot-200.jsonl") for d in ("administrative", "civil", "criminal")] + [
    ("crossdomain-60", ROOT / "data/eval/targets/crossdomain-60/target.jsonl")]
CORPORA = ("administrative-v1", "civil-v1", "criminal-v1", "ip-v1")
PROBE = 60

app = typer.Typer(add_completion=False)

SYSTEM = """당신은 한국 법률 답변을 채점하는 평가자입니다. 질문, 참고 답변, 평가할 답변이 주어집니다.
참고 답변은 한 개의 판례나 법령을 바탕으로 쓴 것입니다. 평가할 답변이 참고 답변의 핵심 결론과 맞는지만 판단하세요.
- 참고 답변에 없는 내용이 더 있어도 감점하지 마세요. 그 내용이 참고 답변과 충돌할 때만 문제로 봅니다.
- 문장 표현이 달라도 법적 결론이 같으면 같은 것으로 봅니다.
- reference_point에는 비교에 쓴 참고 답변의 핵심 결론을 참고 답변에서 그대로 인용하세요.
- covers: 평가할 답변이 그 핵심 결론을 말하면 "yes", 일부만 말하거나 모호하면 "partial", 말하지 않으면 "no".
- contradicts: 평가할 답변에 참고 답변의 결론과 반대되거나 양립할 수 없는 주장이 있으면 true.
- reason에는 판단 근거를 한두 문장으로 쓰세요."""

USER = "질문:\n{question}\n\n참고 답변:\n{reference}\n\n평가할 답변:\n{answer}"


class AnswerGrade(BaseModel):
    reference_point: str = Field(description="참고 답변의 핵심 결론, 참고 답변에서 그대로 인용")
    covers: Literal["yes", "partial", "no"]
    contradicts: bool
    reason: str


def _jsonl(path: Path) -> list[dict]:
    from legal_rag.evaluation.answers.answer_run import read_jsonl
    return read_jsonl(path)


def _evidence_by_key(run: Path) -> dict[str, list[str]]:
    from legal_rag.evaluation.answers.answer_run import evidence_by_key
    try:
        return evidence_by_key(run)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _version(rubric: str) -> str:
    if rubric not in ("core", "v2", "v3", "legacy"):
        raise typer.BadParameter("Rubric must be core, v2, v3 or legacy.")
    return rubric


def _retry_manifest_hashes(run: Path) -> dict[str, str]:
    from legal_rag.evaluation.answers.answer_run import retry_manifest_hashes
    return retry_manifest_hashes(run)


@app.command()
def pairs(run: Path = typer.Option(..., exists=True, file_okay=False),
          qa_root: Path = typer.Option(QA_ROOT, exists=True, file_okay=False)) -> None:
    """Write answer_pairs.jsonl next to verify.jsonl."""
    references: dict[str, list[dict]] = collections.defaultdict(list)
    for path in qa_root.glob("*/qa.jsonl"):
        for record in _jsonl(path):
            references[record["question"].strip()].append(record)
    questions = {f"{name}:{item['id']}": item for name, path in SOURCES for item in _jsonl(path)}
    chunk_documents: dict[str, str] = {}
    for corpus in CORPORA:
        with open(ROOT / f"data/processed/{corpus}/chunks.jsonl", encoding="utf-8") as handle:
            for line in handle:
                chunk = json.loads(line)
                chunk_documents.setdefault(chunk["text"][:PROBE], chunk["document_id"])

    def document_of(evidence: str) -> str | None:
        # An evidence string is its header values, a space, then the chunk text.
        for i in range(min(len(evidence), 300)):
            if (i == 0 or evidence[i - 1] == " ") and evidence[i:i + PROBE] in chunk_documents:
                return chunk_documents[evidence[i:i + PROBE]]
        return None

    out = []
    for turn in _jsonl(run / "verify.jsonl"):
        item = questions[turn["key"]]
        question = item["query"].strip()
        candidates = references.get(question, [])
        candidates = [c for c in candidates if c["document_id"] in item["relevant_document_ids"]] or candidates
        if not candidates:
            typer.echo(f"no reference for {turn['key']}", err=True)
            continue
        documents = [document_of(e) for e in turn.get("evidence", [])]
        out.append({"key": turn["key"], "dataset": turn["dataset"], "kind": turn["kind"], "question": question,
                    "reference": candidates[0]["answer"], "answer": turn.get("final_text", ""),
                    **({"evidence": turn["evidence"]} if "evidence" in turn else {}),
                    "gold": item["relevant_document_ids"], "evidence_documents": documents,
                    "gold_in_evidence": any(d in item["relevant_document_ids"] for d in documents if d)})
    (run / "answer_pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    typer.echo(f"{len(out)} pairs, gold document in evidence for {sum(r['gold_in_evidence'] for r in out)}")


@app.command()
def judge(run: Path = typer.Option(..., exists=True, file_okay=False),
          model: str = typer.Option("qwen3:8b"), base_url: str = typer.Option("http://localhost:11434"),
          think: bool | None = typer.Option(None, "--think/--no-think"),
          num_ctx: int = typer.Option(None, min=1, help="Default 8192 for core, 12288 otherwise."),
          rubric: str = typer.Option("core", help="core checks the reference's conclusion (default since 2026-10-06); "
                                                  "v2, v3 and legacy preserve historical rubrics. v2's grounded score mostly "
                                                  "counts its quote-repair retries; see answer_judge_core.py."),
          max_tokens: int = typer.Option(4096, min=1),
          conclusions: Path = typer.Option(ROOT / "data/eval/judge_cache/core_conclusions.json",
                                           help="core only: shared file of reference conclusions, so every run is measured against the same sentence.")) -> None:
    """Grade saved pairs. core writes answer_judgments-core.jsonl and resumes; v2 uses separate artifacts and retries unresolved grades."""
    from legal_rag.llm import Message, OllamaChatModel, StructuredOutputError

    _version(rubric)
    think = rubric == "legacy" if think is None else think
    if rubric == "core":
        from legal_rag.evaluation.answers.answer_judge_core import JudgeStopped, judge_core_answers

        chat = OllamaChatModel(model=model, base_url=base_url, timeout=240.0, think=think, num_ctx=num_ctx or 8192)
        try:
            summary = judge_core_answers(run, chat, conclusions, emit=typer.echo)
        except JudgeStopped as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"conclusion agreement {summary['conclusion_correct']}/{summary['pairs']} (graded {summary['graded']})")
        return
    num_ctx = num_ctx or 12288
    if rubric in ("v2", "v3"):
        _judge_versioned(run, model, base_url, think, num_ctx, max_tokens, rubric)
        return
    path = run / "answer_judgments.jsonl"
    done = {r["key"] for r in _jsonl(path)} if path.exists() else set()
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=600.0, think=think, num_ctx=num_ctx)
    with path.open("a", encoding="utf-8") as handle:
        for pair in _jsonl(run / "answer_pairs.jsonl"):
            if pair["key"] in done:
                continue
            if not pair["answer"]:
                record = {"key": pair["key"], "grade": None, "error": f"no answer ({pair['kind']})"}
            else:
                messages = [Message(role="system", content=SYSTEM),
                            Message(role="user", content=USER.format(**pair))]
                try:
                    response = chat.generate(messages, response_schema=AnswerGrade, temperature=0.0, max_tokens=max_tokens)
                    record = {"key": pair["key"], "grade": response.parsed.model_dump(), "error": None}
                except StructuredOutputError as exc:
                    record = {"key": pair["key"], "grade": None, "error": str(exc)[:300]}
            record.update(model=chat.name, think=think, num_ctx=num_ctx)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def _judge_versioned(run: Path, model: str, base_url: str, think: bool, num_ctx: int,
                     max_tokens: int, rubric: str = "v2") -> None:
    from legal_rag.evaluation.answers.answer_judge_runner import (
        JudgeOptions, JudgeConfigurationError, JudgeStoppedError, judge_saved_answers,
    )
    from legal_rag.evaluation.answers.answer_rubrics import get_answer_rubric

    options = JudgeOptions(model, base_url, think, num_ctx, max_tokens)
    try:
        judge_saved_answers(run, get_answer_rubric(rubric), options, emit=typer.echo)
    except JudgeConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except JudgeStoppedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _ci(k: int, n: int) -> str:
    """Wilson 95% interval."""
    if not n:
        return "-"
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} [{centre - half:.2f}, {centre + half:.2f}]"


def _ac1(a: list[int], b: list[int]) -> float | None:
    from legal_rag.evaluation.relevance.calibration import _gwet_ac1
    return _gwet_ac1(a, b)


def _correct(grade: dict) -> int:
    return int(grade["covers"] == "yes" and not grade["contradicts"])


@app.command()
def report(run: Path = typer.Option(..., exists=True, file_okay=False),
           manual: Path = typer.Option(None, exists=True, dir_okay=False,
                                       help="Completed review export for v2 (with input fingerprints); legacy accepts {key, covers, contradicts}."),
           rubric: str = typer.Option("v2", help="Select v3, v2 or the historical legacy report.")) -> None:
    _version(rubric)
    if rubric in ("v2", "v3"):
        _report_versioned(run, manual, rubric)
        return
    pairs_by_key = {p["key"]: p for p in _jsonl(run / "answer_pairs.jsonl")}
    grades = {r["key"]: r["grade"] for r in _jsonl(run / "answer_judgments.jsonl") if r["grade"]}
    failed = len(pairs_by_key) - len(grades)
    typer.echo(f"graded {len(grades)}/{len(pairs_by_key)} (unparsed or no answer: {failed})")
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for key, grade in grades.items():
        pair = pairs_by_key[key]
        groups["ALL"].append(grade)
        groups["gold in evidence" if pair["gold_in_evidence"] else "gold not in evidence"].append(grade)
        groups[pair["dataset"]].append(grade)
    typer.echo(f"{'group':24} {'correct (yes, no contradiction)':34} {'covers yes or partial':30} contradicts")
    for name, members in groups.items():
        n = len(members)
        typer.echo(f"{name:24} {_ci(sum(map(_correct, members)), n):34} "
                   f"{_ci(sum(g['covers'] != 'no' for g in members), n):30} "
                   f"{_ci(sum(g['contradicts'] for g in members), n)}")
    if manual:
        hand = {r["key"]: r for r in _jsonl(manual)}
        shared = [k for k in hand if k in grades]
        for label, pick in (("correct", _correct), ("contradicts", lambda g: int(g["contradicts"]))):
            a = [pick(hand[k]) for k in shared]
            b = [pick(grades[k]) for k in shared]
            agree = sum(x == y for x, y in zip(a, b))
            typer.echo(f"agreement with hand grades on {label}: {agree}/{len(shared)}, "
                       f"Gwet AC1 {_ac1(a, b):.3f}; hand rate {sum(a)}/{len(a)}, judge rate {sum(b)}/{len(b)}")


def _report_versioned(run: Path, manual: Path | None, rubric: str = "v2") -> None:
    from legal_rag.evaluation.answers.answer_run import load_answer_run
    from legal_rag.evaluation.answers.answer_rubrics import get_answer_rubric

    rules = get_answer_rubric(rubric).module
    input_fingerprint, metrics = rules.input_fingerprint, rules.metrics
    try:
        saved = load_answer_run(run, rubric, strict=False)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    pairs, evidence, records = saved.pairs, saved.evidence, saved.records
    grades, no_answer, unresolved = saved.grades, saved.no_answer_keys, saved.unresolved_keys
    typer.echo(f"{rubric}: {len(grades)}/{len(pairs)} valid grades; {len(no_answer)} no-answer outcomes; "
               f"{len(unresolved)} unresolved/missing/stale grades or generation errors")
    policies = collections.Counter(records[k].get("retry_policy", "original-identical-retry") for k in grades)
    typer.echo(f"accepted grade retry policies: {dict(policies)}")
    groups = collections.defaultdict(list)
    for key, pair in pairs.items():
        groups["ALL"].append(key)
        groups[pair["dataset"]].append(key)
    for name, keys in groups.items():
        resolved = [k for k in keys if k in grades or k in no_answer]
        unknown = len(keys) - len(resolved)
        typer.echo(f"\n{name}: {len(resolved)}/{len(keys)} resolved")
        for label in ("conclusion_correct", "reference_agreement", "grounded_correct"):
            known = [metrics(grades[k])[label] if k in grades else False for k in resolved]
            available = [v for v in known if v is not None]
            missing_support = len(known) - len(available)
            k = sum(available)
            typer.echo(f"  {label}: {_ci(k, len(available))}; "
                       f"all-case bounds {k}/{len(keys)} to {k + unknown + missing_support}/{len(keys)}")
        members = [grades[k] for k in keys if k in grades]
        typer.echo(f"  basis_match: {dict(collections.Counter(g['basis_match'] for g in members))}")
        typer.echo(f"  evidence_support: {dict(collections.Counter(g['evidence_support'] for g in members))}")
    typer.echo("These scores measure reference agreement and supplied-evidence support, not legal correctness. "
               "Quote checks verify text presence, not entailment. Human calibration remains required.")
    if manual:
        hand = {}
        allowed = {"covers": {"yes", "partial", "no"},
                   "basis_match": {"match", "mismatch", "missing", "not_applicable"},
                   "evidence_support": {"supported", "partial", "unsupported", "not_available"}}
        for row in _jsonl(manual):
            for field, choices in allowed.items():
                if row.get(field) is not None and row[field] not in choices:
                    raise typer.BadParameter(f"Invalid manual {field} for {row['key']}")
            if row.get("contradicts") is not None and not isinstance(row["contradicts"], bool):
                raise typer.BadParameter(f"Manual contradicts must be boolean for {row['key']}")
            if row.get("covers") is None or row.get("contradicts") is None:
                continue
            key = row["key"]
            if key not in pairs:
                continue
            expected = input_fingerprint(pairs[key], evidence.get(key))
            if row.get("input_sha256") != expected:
                raise typer.BadParameter(f"Missing or stale manual input fingerprint for {key}; export a fresh review.")
            if all(field in row for field in ("question", "reference", "answer", "evidence", "kind")):
                if input_fingerprint(row, row["evidence"]) != expected:
                    raise typer.BadParameter(f"Manual review inputs changed for {key}")
            hand[key] = row
        for label in ("conclusion_correct", "reference_agreement", "grounded_correct"):
            needed = ("covers", "contradicts") + (("basis_match",) if label != "conclusion_correct" else ())
            needed += ("evidence_support",) if label == "grounded_correct" else ()
            shared = [k for k in grades if k in hand and all(hand[k].get(f) is not None for f in needed)
                      and metrics(grades[k])[label] is not None]
            if label == "grounded_correct":
                shared = [k for k in shared if hand[k]["evidence_support"] != "not_available"]
            if not shared:
                typer.echo(f"manual agreement on {label}: no applicable completed labels")
                continue
            def pick(row):
                conclusion = bool(_correct(row))
                basis = row.get("basis_match") in ("match", "not_applicable")
                return int(conclusion if label == "conclusion_correct" else conclusion and basis
                           and (label != "grounded_correct" or row["evidence_support"] == "supported"))
            a, b = [pick(hand[k]) for k in shared], [pick(grades[k]) for k in shared]
            ac1 = _ac1(a, b)
            typer.echo(f"manual agreement on {label}: {sum(x == y for x, y in zip(a, b))}/{len(shared)}, "
                       f"Gwet AC1 {'-' if ac1 is None else f'{ac1:.3f}'}")


@app.command()
def review(run: Path = typer.Option(..., exists=True, file_okay=False),
           out: Path = typer.Option(..., help="New JSONL for independent manual grading; does not contain AI grades.")) -> None:
    """Export saved inputs with empty manual labels, without provider identity or AI grades."""
    evidence = _evidence_by_key(run)
    pairs = _jsonl(run / "answer_pairs.jsonl")
    from legal_rag.evaluation.answers.answer_judge import input_fingerprint
    rows = [{"key": p["key"], "kind": p["kind"], "question": p["question"], "reference": p["reference"],
             "answer": p["answer"], "evidence": evidence.get(p["key"], p.get("evidence")),
             "input_sha256": input_fingerprint(p, evidence.get(p["key"], p.get("evidence"))),
             "covers": None, "contradicts": None, "basis_match": None,
             "evidence_support": None, "reason": ""} for p in pairs]
    with out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"{len(rows)} review records. Apply the v2 rubric; fill the manual label fields before reporting agreement.")


if __name__ == "__main__":
    app()
