"""Read and validate saved answer evaluations without invoking a model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .answer_rubrics import get_answer_rubric

METRICS = ("conclusion_correct", "reference_agreement", "grounded_correct")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object.")
            rows.append(row)
    return rows


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_by_key(run: Path) -> dict[str, list[str]]:
    """Preserve saved evidence precedence: verify overrides pairs; contexts fill gaps."""
    evidence = {}
    for name in ("answer_pairs.jsonl", "verify.jsonl"):
        path = run / name
        if path.exists():
            for row in read_jsonl(path):
                if isinstance(row.get("evidence"), list):
                    evidence[row["key"]] = row["evidence"]
    contexts = run / "contexts.jsonl"
    if contexts.exists():
        from legal_rag.schemas import EvidenceBundle
        for row in read_jsonl(contexts):
            if row.get("key") in evidence or not isinstance(row.get("evidence"), dict):
                continue
            try:
                bundle = EvidenceBundle.model_validate(row["evidence"])
            except ValueError as exc:
                raise ValueError(f"Invalid saved evidence bundle for {row.get('key')}") from exc
            evidence[row["key"]] = [" ".join(map(str, item.header.values())) + " " + item.text
                                     for item in bundle.items]
    if any(not isinstance(text, str) for sources in evidence.values() for text in sources):
        raise ValueError("Saved evidence must contain source strings.")
    return evidence


def retry_manifest_hashes(run: Path) -> dict[str, str]:
    hashes = {}
    for path in run.glob("answer_judge_retry-*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes[manifest["policy"]] = canonical_hash(manifest)
    return hashes


def validate_saved_record(pair, evidence, record, manifest_hash, policies, module, schema):
    """Return a validated grade, or None for a valid no-answer outcome."""
    if record.get("manifest_sha256") != manifest_hash:
        raise ValueError("Missing or stale judge manifest fingerprint.")
    if record.get("input_sha256") != module.input_fingerprint(pair, evidence):
        raise ValueError("Missing or stale input fingerprint.")
    policy = record.get("retry_policy")
    if policy and (policy not in policies or record.get("retry_manifest_sha256") != policies[policy]):
        raise ValueError("Invalid retry policy provenance.")
    if record.get("status") == "no_answer" and not pair["answer"].strip() and pair["kind"] != "error":
        return None
    if record.get("status") != "graded" or not record.get("grade"):
        raise ValueError("Unresolved judgment. Rerun judge to retry.")
    grade = schema.model_validate(record["grade"])
    module.validate_quotes(grade, pair, evidence)
    return grade.model_dump()


@dataclass(frozen=True)
class AnswerRun:
    directory: Path
    rubric: str
    pairs: dict
    records: dict
    grades: dict
    scores: dict
    manifest: dict
    evidence: dict
    no_answer_keys: set[str]
    unresolved_keys: set[str]


def load_answer_run(directory: Path, rubric: str, *, strict: bool) -> AnswerRun:
    selected = get_answer_rubric(rubric)
    module, schema = selected.module, selected.grade_schema
    pairs = {row["key"]: row for row in read_jsonl(directory / "answer_pairs.jsonl")}
    evidence = evidence_by_key(directory)
    manifest_path = directory / f"answer_judge-{rubric}.json"
    judgment_path = directory / f"answer_judgments-{rubric}.jsonl"
    if not manifest_path.exists() or not judgment_path.exists():
        raise ValueError(f"{directory}/{rubric}: no judgments. Run judge with --rubric {rubric} first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = canonical_hash(manifest)
    records = {row["key"]: row for row in read_jsonl(judgment_path)}
    policies = retry_manifest_hashes(directory)
    grades, scores, no_answer, unresolved = {}, {}, set(), set()
    for key, pair in pairs.items():
        try:
            grade = validate_saved_record(pair, evidence.get(key), records.get(key, {}),
                                          fingerprint, policies, module, schema)
        except (ValueError, KeyError) as exc:
            if strict:
                raise ValueError(f"{directory}/{rubric}: {key}: {exc}") from exc
            unresolved.add(key)
            continue
        if grade is None:
            no_answer.add(key)
            scores[key] = dict.fromkeys(METRICS, False)
        else:
            grades[key] = grade
            scores[key] = module.metrics(grade)
    return AnswerRun(directory, rubric, pairs, records, grades, scores, manifest,
                     evidence, no_answer, unresolved)
