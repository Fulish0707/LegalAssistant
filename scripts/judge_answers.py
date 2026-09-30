#!/usr/bin/env python
"""Grade saved /v1/chat answers against the AIHub reference answers.

`pairs` joins a verify.jsonl from eval_chat_pipeline.py with the reference answers in the
upstream qa.jsonl files (the eval sets leave them out) and marks whether the question's gold
document was among the turn's evidence. The references are written from that one document, so
an answer built on other evidence can be sound and still miss them; report the two groups apart.
`judge` asks an Ollama model two questions per pair: does the answer state the reference's key
conclusion (yes / partial / no), and does anything in it contradict the reference.
`report` prints both rates by group and dataset, and agreement with hand grades if given.

This measures agreement with the reference answers, not legal correctness.

Example (one heavy process at a time; unload other Ollama models first):
    uv run python scripts/judge_answers.py pairs --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles
    uv run python scripts/judge_answers.py judge --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles
    uv run python scripts/judge_answers.py report --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles \
        --manual data/eval/reports/chat_pipeline/formal-2026-09-28-articles/manual_grades.jsonl
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
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


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
                    "gold": item["relevant_document_ids"], "evidence_documents": documents,
                    "gold_in_evidence": any(d in item["relevant_document_ids"] for d in documents if d)})
    (run / "answer_pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    typer.echo(f"{len(out)} pairs, gold document in evidence for {sum(r['gold_in_evidence'] for r in out)}")


@app.command()
def judge(run: Path = typer.Option(..., exists=True, file_okay=False),
          model: str = typer.Option("qwen3:8b"), base_url: str = typer.Option("http://localhost:11434"),
          think: bool = typer.Option(True, "--think/--no-think"), num_ctx: int = typer.Option(12288)) -> None:
    """Append one grade per pair to answer_judgments.jsonl; safe to interrupt and rerun."""
    from legal_rag.llm import Message, OllamaChatModel, StructuredOutputError

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
                    response = chat.generate(messages, response_schema=AnswerGrade, temperature=0.0, max_tokens=4096)
                    record = {"key": pair["key"], "grade": response.parsed.model_dump(), "error": None}
                except StructuredOutputError as exc:
                    record = {"key": pair["key"], "grade": None, "error": str(exc)[:300]}
            record.update(model=chat.name, think=think, num_ctx=num_ctx)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def _ci(k: int, n: int) -> str:
    """Wilson 95% interval."""
    if not n:
        return "-"
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} [{centre - half:.2f}, {centre + half:.2f}]"


def _ac1(a: list[int], b: list[int]) -> float | None:
    from legal_rag.evaluation.calibration import _gwet_ac1
    return _gwet_ac1(a, b)


def _correct(grade: dict) -> int:
    return int(grade["covers"] == "yes" and not grade["contradicts"])


@app.command()
def report(run: Path = typer.Option(..., exists=True, file_okay=False),
           manual: Path = typer.Option(None, exists=True, dir_okay=False,
                                       help="Hand grades: {key, covers, contradicts} per line.")) -> None:
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


if __name__ == "__main__":
    app()
