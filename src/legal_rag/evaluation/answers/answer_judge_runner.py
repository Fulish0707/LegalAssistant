"""Shared lifecycle for versioned saved-answer judges."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

from legal_rag.llm import Message, OllamaChatModel, StructuredOutputError
from .answer_judge import (SPAN_CATALOG_VERSION, MAX_EVIDENCE_SELECTIONS,
                           JUDGE_TIMEOUT_SECONDS, TIMEOUT_STOP_AFTER, QuoteValidationError)
from .answer_rubrics import AnswerRubric
from .answer_run import evidence_by_key, read_jsonl, retry_manifest_hashes, validate_saved_record
from .spans import SPAN_MAX_CHARS
from ..relevance.relevance_judge import ollama_model_digest


@dataclass(frozen=True)
class JudgeOptions:
    model: str
    base_url: str
    think: bool
    num_ctx: int
    max_tokens: int


class JudgeConfigurationError(ValueError):
    """Settings or saved provenance prevent a safe grading run."""


class JudgeStoppedError(RuntimeError):
    """An operational stop whose partial results have already been saved."""


def build_judge_manifest(rubric: AnswerRubric, options: JudgeOptions, digest: str) -> dict:
    rules = rubric.module
    model, base_url, think = options.model, options.base_url, options.think
    num_ctx, max_tokens = options.num_ctx, options.max_tokens
    manifest = {"rubric": rules.RUBRIC_VERSION, "system": rules.SYSTEM,
                "schema": rubric.grade_schema.model_json_schema(), "quote_rule_version": rules.QUOTE_RULE_VERSION,
                "model": f"ollama:{model}", "model_digest": digest, "base_url": base_url,
                "think": think, "num_ctx": num_ctx, "max_tokens": max_tokens, "temperature": 0.0,
                "input_format": "question/reference/answer/evidence JSON; no provider or gold IDs",
                "answer_text_policy": "saved answer_pairs.answer; no automatic truncation",
                "attempts_per_invocation": 2}
    return manifest


def build_retry_manifest(rubric: AnswerRubric) -> dict:
    rules = rubric.module
    retry_manifest = {"policy": rules.RETRY_POLICY_VERSION, "selection_instruction": rules.SELECTION_INSTRUCTION,
                      "span_catalog_version": SPAN_CATALOG_VERSION, "max_span_chars": SPAN_MAX_CHARS,
                      "selection_schema": rules.SelectionGrade.model_json_schema(),
                      "strategy": "initial v2 quote output; known or new schema/quote failures repair by constrained passage IDs",
                      "attempts_per_invocation": 2, "quote_rule_version": rules.QUOTE_RULE_VERSION,
                      "max_evidence_selections": MAX_EVIDENCE_SELECTIONS, "unique_evidence_ids": True,
                      "request_timeout_seconds": JUDGE_TIMEOUT_SECONDS,
                      "timeout_retry": False, "consecutive_timeout_stop_after": TIMEOUT_STOP_AFTER}
    if rubric.name == "v3":
        retry_manifest.update(strategy="status-dependent passage IDs; basis match restricted to citation-containing answer passages; summary labels derived from statuses",
                              citation_presence=True, status_dependent_selection=True,
                              checklist_limit="max(8, 2 * number of reference passages)",
                              unique_checklist_items=True,
                              server_http_5xx_stop=True)
    return retry_manifest


def judge_saved_answers(run: Path, rubric: AnswerRubric, options: JudgeOptions,
                        *, emit: Callable[[str], None]) -> None:
    rules = rubric.module
    model, base_url, think = options.model, options.base_url, options.think
    num_ctx, max_tokens = options.num_ctx, options.max_tokens
    pairs = {p["key"]: p for p in read_jsonl(run / "answer_pairs.jsonl")}
    try:
        evidence = evidence_by_key(run)
    except ValueError as exc:
        raise JudgeConfigurationError(str(exc)) from exc
    path, manifest_path = run / f"answer_judgments-{rubric.name}.jsonl", run / f"answer_judge-{rubric.name}.json"
    previous = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    attempts_path = run / f"answer_judge_attempts-{rubric.name}.jsonl"
    previous_attempts = {(r["key"], r.get("invocation_id")): r for r in read_jsonl(attempts_path)} if attempts_path.exists() else {}
    digest = ollama_model_digest(base_url, model)
    if not digest:
        raise JudgeConfigurationError("Cannot identify the local judge model digest; check Ollama before judging.")
    manifest = build_judge_manifest(rubric, options, digest)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise JudgeConfigurationError("Judge configuration changed. Use a separate run directory for new settings.")
    elif previous:
        raise JudgeConfigurationError(f"Existing {rubric.name} grades have no manifest; cannot safely resume.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Retry strategy is supplemental provenance: the base rubric/settings remain
    # immutable and already validated grades keep their original history.
    retry_manifest = build_retry_manifest(rubric)
    retry_hash = hashlib.sha256(json.dumps(retry_manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    retry_path = run / f"answer_judge_retry-{rules.RETRY_POLICY_VERSION}.json"
    if retry_path.exists():
        if json.loads(retry_path.read_text()) != retry_manifest:
            raise JudgeConfigurationError("Retry policy changed; bump its version before resuming.")
    else:
        retry_path.write_text(json.dumps(retry_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retry_manifests = retry_manifest_hashes(run)
    invocation = uuid.uuid4().hex
    consecutive_timeouts = 0
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=JUDGE_TIMEOUT_SECONDS, think=think, num_ctx=num_ctx)
    try:
        with path.open("a", encoding="utf-8") as handle, (run / f"answer_judge_attempts-{rubric.name}.jsonl").open("a", encoding="utf-8") as diagnostics:
            for key, pair in pairs.items():
                sources = evidence.get(key)
                fingerprint = rules.input_fingerprint(pair, sources)
                old = previous.get(key, {})
                try:
                    validate_saved_record(pair, sources, old, manifest_hash, retry_manifests,
                                          rules, rubric.grade_schema)
                except (ValueError, KeyError):
                    pass
                else:
                    continue
                record = {"key": key, "rubric": rules.RUBRIC_VERSION, "model": chat.name,
                          "input_sha256": fingerprint, "manifest_sha256": manifest_hash,
                          "retry_policy": rules.RETRY_POLICY_VERSION, "retry_manifest_sha256": retry_hash,
                          "invocation_id": invocation,
                          "evidence_available": sources is not None}
                timed_out = False
                server_error = False
                if not pair["answer"].strip():
                    record.update(status="generation_error" if pair["kind"] == "error" else "no_answer",
                                  grade=None, error=f"no answer ({pair['kind']})")
                else:
                    messages = [Message(role="system", content=rules.SYSTEM),
                                Message(role="user", content=rules.messages_content(pair, sources))]
                    repair_via_selection = (old.get("status") == "error"
                                            and old.get("input_sha256") == fingerprint
                                            and old.get("manifest_sha256") == manifest_hash
                                            and old.get("error", "").startswith(("ValueError:", "QuoteValidationError:", "StructuredOutputError:"))
                                            or (old.get("status") == "error"
                                                and old.get("input_sha256") == fingerprint
                                                and old.get("manifest_sha256") == manifest_hash
                                                and (old.get("selection_mode") is True
                                                     or previous_attempts.get((key, old.get("invocation_id")), {}).get("selection_catalog") is not None)))
                    repair_via_selection = rubric.initial_selection or repair_via_selection
                    selection_errors = []
                    for attempt in range(1, 3):
                        record["attempts"] = attempt
                        started = time.perf_counter()
                        detail = {"key": key, "attempt": attempt, "invocation_id": invocation,
                                  "input_sha256": fingerprint, "manifest_sha256": manifest_hash,
                                  "retry_policy": rules.RETRY_POLICY_VERSION, "retry_manifest_sha256": retry_hash,
                                  "raw_output": None, "parsed_grade": None,
                                  "validation_errors": [], "repair_feedback": None}
                        try:
                            record["selection_mode"] = repair_via_selection
                            detail["selection_mode"] = repair_via_selection
                            schema, attempt_messages = rubric.grade_schema, messages
                            catalog = None
                            if repair_via_selection:
                                schema, catalog, content = rules.selection_request(pair, sources)
                                if selection_errors:
                                    content += "\n\n이전 ID 채점의 검증 오류를 원문과 같은 평가 기준에 따라 수정하세요:\n" + json.dumps(selection_errors, ensure_ascii=False)
                                attempt_messages = [Message(role="system", content=rules.SYSTEM + "\n\n" + rules.SELECTION_INSTRUCTION),
                                                    Message(role="user", content=content)]
                                detail.update(selection_catalog=catalog, repair_feedback=rules.SELECTION_INSTRUCTION,
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
                                parsed = rules.materialize_selection(parsed, catalog)
                            detail["parsed_grade"] = parsed.model_dump()
                            rules.validate_quotes(parsed, pair, sources)
                            grade = parsed.model_dump()
                            record.update(status="graded", grade=grade, metrics=rules.metrics(grade), error=None)
                        except httpx.TimeoutException as exc:
                            timed_out = True
                            record.update(status="error", grade=None,
                                          error=f"{type(exc).__name__}: {str(exc)[:300]}",
                                          timeout_seconds=JUDGE_TIMEOUT_SECONDS)
                        except httpx.HTTPStatusError as exc:
                            server_error = exc.response.status_code >= 500
                            server_message = exc.response.text[:1500]
                            detail["http_response_body"] = server_message
                            record.update(status="error", grade=None,
                                          error=f"Judge HTTP {exc.response.status_code}: {server_message}")
                        except (StructuredOutputError, ValueError, httpx.TransportError) as exc:
                            if isinstance(exc, StructuredOutputError):
                                detail["raw_output"] = exc.text
                            if isinstance(exc, (StructuredOutputError, ValueError)):
                                detail["repair_feedback"] = rules.SELECTION_INSTRUCTION
                                detail["validation_errors"] = exc.issues if isinstance(exc, QuoteValidationError) else [
                                    {"field": "output_json", "message": str(exc)[:1000]}]
                                if attempt < 2:
                                    repair_via_selection = True
                                    selection_errors = detail["validation_errors"]
                                    if rubric.initial_selection and catalog is not None:
                                        selection_errors = rules.selection_feedback(
                                            selection_errors, catalog, detail.get("selected_grade"))
                            record.update(status="error", grade=None, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                        detail.update(status=record["status"], error=record["error"],
                                      seconds=round(time.perf_counter() - started, 3))
                        diagnostics.write(json.dumps(detail, ensure_ascii=False) + "\n")
                        diagnostics.flush()
                        if record["status"] == "graded" or timed_out or server_error:
                            break
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                emit(f"{key}: {record['status']}")
                if server_error:
                    raise JudgeStoppedError(
                        f"Stopped on judge server failure: {record['error']}\n"
                        "Saved results are retained. Resolve the server error before resuming; "
                        "memory errors may require a smaller --num-ctx in a separate run.")
                consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0
                if consecutive_timeouts >= TIMEOUT_STOP_AFTER:
                    raise JudgeStoppedError(
                        "Stopped after consecutive judge timeouts. Saved results are retained; "
                        "check Ollama responsiveness and grading diagnostics before resuming.")
    finally:
        chat._client.close()
