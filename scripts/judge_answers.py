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
import hashlib
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


def _evidence_by_key(run: Path) -> dict[str, list[str]]:
    """Use the saved generator evidence, never the gold/reference document as evidence."""
    evidence = {}
    pair_path = run / "answer_pairs.jsonl"
    if pair_path.exists():
        for row in _jsonl(pair_path):
            if isinstance(row.get("evidence"), list):
                evidence[row["key"]] = row["evidence"]
    verify = run / "verify.jsonl"
    if verify.exists():
        for row in _jsonl(verify):
            if "evidence" in row and isinstance(row["evidence"], list):
                evidence[row["key"]] = row["evidence"]
    contexts = run / "contexts.jsonl"
    if contexts.exists():
        for row in _jsonl(contexts):
            if row.get("key") in evidence or not isinstance(row.get("evidence"), dict):
                continue
            from legal_rag.schemas import EvidenceBundle
            try:
                bundle = EvidenceBundle.model_validate(row["evidence"])
            except ValueError as exc:
                raise typer.BadParameter(f"Invalid saved evidence bundle for {row.get('key')}") from exc
            evidence[row["key"]] = [" ".join(map(str, item.header.values())) + " " + item.text
                                     for item in bundle.items]
    if any(not isinstance(text, str) for sources in evidence.values() for text in sources):
        raise typer.BadParameter("Saved evidence must contain source strings.")
    return evidence


def _version(rubric: str) -> str:
    if rubric not in ("v2", "legacy"):
        raise typer.BadParameter("Rubric must be v2 or legacy.")
    return rubric


def _retry_manifest_hashes(run: Path) -> dict[str, str]:
    hashes = {}
    for path in run.glob("answer_judge_retry-*.json"):
        manifest = json.loads(path.read_text())
        hashes[manifest["policy"]] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return hashes


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
          think: bool | None = typer.Option(None, "--think/--no-think"), num_ctx: int = typer.Option(12288, min=1),
          rubric: str = typer.Option("v2", help="v2 separates conclusion, basis and evidence; legacy preserves the old rubric."),
          max_tokens: int = typer.Option(4096, min=1)) -> None:
    """Grade saved pairs; v2 uses separate artifacts and retries unresolved grades."""
    from legal_rag.llm import Message, OllamaChatModel, StructuredOutputError

    _version(rubric)
    think = rubric == "legacy" if think is None else think
    if rubric == "v2":
        _judge_v2(run, model, base_url, think, num_ctx, max_tokens)
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


def _judge_v2(run: Path, model: str, base_url: str, think: bool, num_ctx: int, max_tokens: int) -> None:
    import time
    import uuid
    import httpx
    from legal_rag.evaluation.answer_judge import (
        AnswerGradeV2, QUOTE_RULE_VERSION, SELECTION_INSTRUCTION, SPAN_CATALOG_VERSION, SPAN_MAX_CHARS, SelectionGrade,
        RETRY_POLICY_VERSION, RUBRIC_VERSION, SYSTEM as V2_SYSTEM,
        MAX_EVIDENCE_SELECTIONS, JUDGE_TIMEOUT_SECONDS, TIMEOUT_STOP_AFTER,
        QuoteValidationError, input_fingerprint, materialize_selection, messages_content,
        metrics, selection_request, validate_quotes)
    from legal_rag.evaluation.relevance_judge import ollama_model_digest
    from legal_rag.llm import Message, OllamaChatModel, StructuredOutputError

    pairs = {p["key"]: p for p in _jsonl(run / "answer_pairs.jsonl")}
    evidence = _evidence_by_key(run)
    for key, pair in pairs.items():
        if key not in evidence and isinstance(pair.get("evidence"), list):
            evidence[key] = pair["evidence"]
    path, manifest_path = run / "answer_judgments-v2.jsonl", run / "answer_judge-v2.json"
    previous = {r["key"]: r for r in _jsonl(path)} if path.exists() else {}
    attempts_path = run / "answer_judge_attempts-v2.jsonl"
    previous_attempts = {(r["key"], r.get("invocation_id")): r for r in _jsonl(attempts_path)} if attempts_path.exists() else {}
    digest = ollama_model_digest(base_url, model)
    if not digest:
        raise typer.BadParameter("Cannot identify the local judge model digest; check Ollama before judging.")
    manifest = {"rubric": RUBRIC_VERSION, "system": V2_SYSTEM,
                "schema": AnswerGradeV2.model_json_schema(), "quote_rule_version": QUOTE_RULE_VERSION,
                "model": f"ollama:{model}", "model_digest": digest, "base_url": base_url,
                "think": think, "num_ctx": num_ctx, "max_tokens": max_tokens, "temperature": 0.0,
                "input_format": "question/reference/answer/evidence JSON; no provider or gold IDs",
                "answer_text_policy": "saved answer_pairs.answer; no automatic truncation",
                "attempts_per_invocation": 2}
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise typer.BadParameter("Judge configuration changed. Use a separate run directory for new settings.")
    elif previous:
        raise typer.BadParameter("Existing v2 grades have no manifest; cannot safely resume.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Retry strategy is supplemental provenance: the base rubric/settings remain
    # immutable and already validated grades keep their original history.
    retry_manifest = {"policy": RETRY_POLICY_VERSION, "selection_instruction": SELECTION_INSTRUCTION,
                      "span_catalog_version": SPAN_CATALOG_VERSION, "max_span_chars": SPAN_MAX_CHARS,
                      "selection_schema": SelectionGrade.model_json_schema(),
                      "strategy": "initial v2 quote output; known or new schema/quote failures repair by constrained passage IDs",
                      "attempts_per_invocation": 2, "quote_rule_version": QUOTE_RULE_VERSION,
                      "max_evidence_selections": MAX_EVIDENCE_SELECTIONS, "unique_evidence_ids": True,
                      "request_timeout_seconds": JUDGE_TIMEOUT_SECONDS,
                      "timeout_retry": False, "consecutive_timeout_stop_after": TIMEOUT_STOP_AFTER}
    retry_hash = hashlib.sha256(json.dumps(retry_manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    retry_path = run / f"answer_judge_retry-{RETRY_POLICY_VERSION}.json"
    if retry_path.exists():
        if json.loads(retry_path.read_text()) != retry_manifest:
            raise typer.BadParameter("Retry policy changed; bump its version before resuming.")
    else:
        retry_path.write_text(json.dumps(retry_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retry_manifests = _retry_manifest_hashes(run)
    invocation = uuid.uuid4().hex
    consecutive_timeouts = 0
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=JUDGE_TIMEOUT_SECONDS, think=think, num_ctx=num_ctx)
    try:
        with path.open("a", encoding="utf-8") as handle, (run / "answer_judge_attempts-v2.jsonl").open("a", encoding="utf-8") as diagnostics:
            for key, pair in pairs.items():
                sources = evidence.get(key)
                fingerprint = input_fingerprint(pair, sources)
                old = previous.get(key, {})
                old_policy = old.get("retry_policy")
                retry_provenance_ok = (not old_policy or
                                       (old_policy in retry_manifests and old.get("retry_manifest_sha256") == retry_manifests[old_policy]))
                if (old.get("input_sha256") == fingerprint and old.get("manifest_sha256") == manifest_hash
                        and retry_provenance_ok
                        and (old.get("grade") is not None or old.get("status") == "no_answer")):
                    if old.get("status") == "no_answer" and not pair["answer"].strip() and pair["kind"] != "error":
                        continue
                    try:
                        saved = AnswerGradeV2.model_validate(old["grade"])
                        validate_quotes(saved, pair, sources)
                    except (ValueError, KeyError):
                        pass
                    else:
                        if old.get("status") == "graded":
                            continue
                record = {"key": key, "rubric": RUBRIC_VERSION, "model": chat.name,
                          "input_sha256": fingerprint, "manifest_sha256": manifest_hash,
                          "retry_policy": RETRY_POLICY_VERSION, "retry_manifest_sha256": retry_hash,
                          "invocation_id": invocation,
                          "evidence_available": sources is not None}
                timed_out = False
                if not pair["answer"].strip():
                    record.update(status="generation_error" if pair["kind"] == "error" else "no_answer",
                                  grade=None, error=f"no answer ({pair['kind']})")
                else:
                    messages = [Message(role="system", content=V2_SYSTEM),
                                Message(role="user", content=messages_content(pair, sources))]
                    repair_via_selection = (old.get("status") == "error"
                                            and old.get("input_sha256") == fingerprint
                                            and old.get("manifest_sha256") == manifest_hash
                                            and old.get("error", "").startswith(("ValueError:", "QuoteValidationError:", "StructuredOutputError:"))
                                            or (old.get("status") == "error"
                                                and old.get("input_sha256") == fingerprint
                                                and old.get("manifest_sha256") == manifest_hash
                                                and (old.get("selection_mode") is True
                                                     or previous_attempts.get((key, old.get("invocation_id")), {}).get("selection_catalog") is not None)))
                    selection_errors = []
                    for attempt in range(1, 3):
                        record["attempts"] = attempt
                        started = time.perf_counter()
                        detail = {"key": key, "attempt": attempt, "invocation_id": invocation,
                                  "input_sha256": fingerprint, "manifest_sha256": manifest_hash,
                                  "retry_policy": RETRY_POLICY_VERSION, "retry_manifest_sha256": retry_hash,
                                  "raw_output": None, "parsed_grade": None,
                                  "validation_errors": [], "repair_feedback": None}
                        try:
                            record["selection_mode"] = repair_via_selection
                            detail["selection_mode"] = repair_via_selection
                            schema, attempt_messages = AnswerGradeV2, messages
                            catalog = None
                            if repair_via_selection:
                                schema, catalog, content = selection_request(pair, sources)
                                if selection_errors:
                                    content += "\n\n이전 ID 채점의 검증 오류를 원문과 같은 평가 기준에 따라 수정하세요:\n" + json.dumps(selection_errors, ensure_ascii=False)
                                attempt_messages = [Message(role="system", content=V2_SYSTEM + "\n\n" + SELECTION_INSTRUCTION),
                                                    Message(role="user", content=content)]
                                detail.update(selection_catalog=catalog, repair_feedback=SELECTION_INSTRUCTION,
                                              previous_validation_errors=selection_errors)
                            detail["output_schema_sha256"] = hashlib.sha256(
                                json.dumps(schema.model_json_schema(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            response = chat.generate(attempt_messages, response_schema=schema,
                                                     temperature=0.0, max_tokens=max_tokens)
                            detail.update(raw_output=response.text,
                                          generation_metadata={field: response.raw.get(field) for field in
                                              ("done_reason", "prompt_eval_count", "eval_count", "total_duration")})
                            parsed = response.parsed
                            if catalog is not None:
                                detail["selected_grade"] = parsed.model_dump(mode="json")
                                parsed = materialize_selection(parsed, catalog)
                            detail["parsed_grade"] = parsed.model_dump()
                            validate_quotes(parsed, pair, sources)
                            grade = parsed.model_dump()
                            record.update(status="graded", grade=grade, metrics=metrics(grade), error=None)
                        except httpx.TimeoutException as exc:
                            timed_out = True
                            record.update(status="error", grade=None,
                                          error=f"{type(exc).__name__}: {str(exc)[:300]}",
                                          timeout_seconds=JUDGE_TIMEOUT_SECONDS)
                        except httpx.HTTPStatusError as exc:
                            record.update(status="error", grade=None, error=f"Judge HTTP {exc.response.status_code}")
                        except (StructuredOutputError, ValueError, httpx.TransportError) as exc:
                            if isinstance(exc, StructuredOutputError):
                                detail["raw_output"] = exc.text
                            if isinstance(exc, (StructuredOutputError, ValueError)):
                                detail["repair_feedback"] = SELECTION_INSTRUCTION
                                detail["validation_errors"] = exc.issues if isinstance(exc, QuoteValidationError) else [
                                    {"field": "output_json", "message": str(exc)[:1000]}]
                                if attempt < 2:
                                    repair_via_selection = True
                                    selection_errors = detail["validation_errors"]
                            record.update(status="error", grade=None, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                        detail.update(status=record["status"], error=record["error"],
                                      seconds=round(time.perf_counter() - started, 3))
                        diagnostics.write(json.dumps(detail, ensure_ascii=False) + "\n")
                        diagnostics.flush()
                        if record["status"] == "graded" or timed_out:
                            break
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                typer.echo(f"{key}: {record['status']}")
                consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0
                if consecutive_timeouts >= TIMEOUT_STOP_AFTER:
                    typer.echo("Stopped after consecutive judge timeouts. Saved results are retained; "
                               "recover Ollama before resuming.", err=True)
                    raise typer.Exit(code=1)
    finally:
        chat._client.close()


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
                                       help="Completed review export for v2 (with input fingerprints); legacy accepts {key, covers, contradicts}."),
           rubric: str = typer.Option("v2", help="Select v2 or the historical legacy report.")) -> None:
    _version(rubric)
    if rubric == "v2":
        _report_v2(run, manual)
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


def _report_v2(run: Path, manual: Path | None) -> None:
    from legal_rag.evaluation.answer_judge import AnswerGradeV2, input_fingerprint, metrics, validate_quotes

    pairs = {p["key"]: p for p in _jsonl(run / "answer_pairs.jsonl")}
    evidence = _evidence_by_key(run)
    for key, pair in pairs.items():
        if key not in evidence and isinstance(pair.get("evidence"), list):
            evidence[key] = pair["evidence"]
    manifest_path, path = run / "answer_judge-v2.json", run / "answer_judgments-v2.jsonl"
    if not manifest_path.exists() or not path.exists():
        raise typer.BadParameter("No v2 judgments. Run judge with --rubric v2 first; legacy scores remain separate.")
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    records = {r["key"]: r for r in _jsonl(path)}
    retry_manifests = _retry_manifest_hashes(run)
    grades, no_answer = {}, set()
    for key, pair in pairs.items():
        row = records.get(key, {})
        if row.get("manifest_sha256") != manifest_hash or row.get("input_sha256") != input_fingerprint(pair, evidence.get(key)):
            continue
        policy = row.get("retry_policy")
        if policy and (policy not in retry_manifests or row.get("retry_manifest_sha256") != retry_manifests[policy]):
            continue
        if row.get("status") == "no_answer" and not pair["answer"].strip() and pair["kind"] != "error":
            no_answer.add(key)
        elif row.get("status") == "graded" and row.get("grade"):
            try:
                grade = AnswerGradeV2.model_validate(row["grade"])
                validate_quotes(grade, pair, evidence.get(key))
            except ValueError:
                continue
            grades[key] = grade.model_dump()
    unresolved = set(pairs) - grades.keys() - no_answer
    typer.echo(f"v2: {len(grades)}/{len(pairs)} valid grades; {len(no_answer)} no-answer outcomes; "
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
    from legal_rag.evaluation.answer_judge import input_fingerprint
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
