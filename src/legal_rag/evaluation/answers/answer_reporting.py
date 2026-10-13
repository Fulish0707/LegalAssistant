"""Paired reports for frozen answers; reporting never invokes generation or retrieval."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import tempfile

from .answer_run import AnswerRun, METRICS, load_answer_run

INTERPRETATION = "Automated reference agreement and supplied-evidence support; not legal correctness or human calibration."
ARMS = ("original", "structural")


def _paired_keys(original: AnswerRun, structural: AnswerRun) -> list[str]:
    keys = list(original.pairs)
    if not keys:
        raise ValueError("Cannot compare empty answer runs.")
    if set(keys) != set(structural.pairs):
        raise ValueError("Arms have different keys.")
    if original.manifest != structural.manifest:
        raise ValueError("Arms used different judge configurations.")
    if original.unresolved_keys or structural.unresolved_keys:
        raise ValueError("Resolve all judgments before paired reporting.")
    for key in keys:
        if any(original.pairs[key].get(field) != structural.pairs[key].get(field)
               for field in ("question", "reference", "dataset", "gold")):
            raise ValueError(f"Paired inputs differ: {key}")
    return keys


def _metric_summary(original: AnswerRun, structural: AnswerRun, keys: list[str]) -> dict:
    result = {}
    for metric in METRICS:
        comparable = [key for key in keys if original.scores[key][metric] is not None
                      and structural.scores[key][metric] is not None]
        left = [original.scores[key][metric] for key in comparable]
        right = [structural.scores[key][metric] for key in comparable]
        delta = [int(b) - int(a) for a, b in zip(left, right)]
        n = len(comparable)
        rng = random.Random(42)
        boot = sorted(sum(rng.choices(delta, k=n)) / n for _ in range(10000)) if n else []
        entry = {"original": sum(left), "structural": sum(right),
                 "difference": sum(delta) / n if n else None,
                 "paired_bootstrap_95pct": [boot[250], boot[9749]] if boot else None,
                 "improved": [key for key, d in zip(comparable, delta) if d == 1],
                 "regressed": [key for key, d in zip(comparable, delta) if d == -1]}
        if n != len(keys):
            entry["paired_resolved"] = n
        result[metric] = entry
    return result


def compare_answer_runs(original: AnswerRun, structural: AnswerRun) -> dict:
    keys = _paired_keys(original, structural)
    summaries = {}
    for name, run in zip(ARMS, (original, structural)):
        grades = list(run.grades.values())
        summaries[name] = {
            "basis_match": dict(Counter(grade["basis_match"] for grade in grades)),
            "evidence_support": dict(Counter(grade["evidence_support"] for grade in grades)),
            "gold_in_evidence": sum(pair.get("gold_in_evidence", False) for pair in run.pairs.values())}
    return {"n": len(keys), "metrics": _metric_summary(original, structural, keys), "arms": summaries}


def _groups(original: AnswerRun) -> dict[str, list[str]]:
    keys = list(original.pairs)
    return {"ALL": keys, **{dataset: [key for key in keys if original.pairs[key]["dataset"] == dataset]
                            for dataset in sorted({row["dataset"] for row in original.pairs.values()})}}


def compare_rubric_versions(old: AnswerRun, new: AnswerRun) -> dict:
    keys = list(old.pairs)
    if set(keys) != set(new.pairs) or old.unresolved_keys or new.unresolved_keys:
        raise ValueError("Rubric comparison requires identical, resolved case sets.")
    for key in keys:
        if any(old.pairs[key].get(field) != new.pairs[key].get(field)
               for field in ("question", "reference", "answer", "dataset", "gold", "kind")):
            raise ValueError(f"Rubric inputs differ: {key}")
        if old.evidence.get(key) != new.evidence.get(key):
            raise ValueError(f"Rubric evidence differs: {key}")
    transitions = {metric: dict(Counter(f"{old.scores[key][metric]} -> {new.scores[key][metric]}" for key in keys))
                   for metric in METRICS}
    changes = [{"key": key, "v2_metrics": old.scores[key], "v3_metrics": new.scores[key],
                "v2_grade": old.grades.get(key), "v3_grade": new.grades.get(key)}
               for key in keys if old.scores[key] != new.scores[key]]
    return {"transitions": transitions, "changes": changes}


def _write_reports(directory: Path, reports: dict[str, str]) -> None:
    for name, content in reports.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=f".{name}.", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            os.replace(temporary, directory / name)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_golden_report(run: Path) -> dict:
    original, structural = (load_answer_run(run / arm, "v2", strict=True) for arm in ARMS)
    summary = compare_answer_runs(original, structural)
    if summary["n"] != 80:
        raise ValueError("Golden-answer runner requires 80 paired questions.")
    generation = [json.loads((run / arm / "generation.json").read_text(encoding="utf-8")) for arm in ARMS]
    settings = [{key: value for key, value in item.items() if key not in ("arm", "contexts_sha256")}
                for item in generation]
    if settings[0] != settings[1]:
        raise ValueError("Generation settings differ between arms.")
    groups = {}
    for dataset, keys in _groups(original).items():
        if dataset == "ALL":
            continue
        groups[dataset] = {"n": len(keys)}
        for metric, item in _metric_summary(original, structural, keys).items():
            groups[dataset][metric] = {"original": item["original"], "structural": item["structural"],
                                       "improved": len(item["improved"]), "regressed": len(item["regressed"])}
            if "paired_resolved" in item:
                groups[dataset][metric]["paired_resolved"] = item["paired_resolved"]
    lines = ["# Paired 80-question golden-answer comparison", "",
             "80 historical questions (20 administrative, civil, criminal and crossdomain each). "
             "Original four-index fanout versus smaller structural chunks; neighbors disabled in both arms. "
             "Both arms use the same generation settings and a 7,500-character maximum rendered context. Full answer text is graded.", "",
             "| Metric | Original | Structural | Change |", "|---|---:|---:|---:|"]
    for metric, item in summary["metrics"].items():
        n = item.get("paired_resolved", summary["n"])
        change = f"{item['difference']:+.0%}" if n else "not available"
        lines.append(f"| {metric} | {item['original']}/{n} | {item['structural']}/{n} | {change} |")
    lines += ["", "## Paired changes", ""]
    for metric, item in summary["metrics"].items():
        interval = item["paired_bootstrap_95pct"]
        bounds = f"[{interval[0]:+.1%}, {interval[1]:+.1%}]" if interval else "not available"
        lines.append(f"- {metric}: {len(item['improved'])} improved, {len(item['regressed'])} regressed; "
                     f"paired bootstrap 95% interval {bounds} (10,000 samples, seed 42).")
    lines += ["", "## Interpretation limits", "", INTERPRETATION,
              "This is an exploratory development sample. Chunk size and structural splitting change together; "
              "actual evidence lengths differ. Human calibration is still required. "
              "Review changed cases in [case-review.md](case-review.md).", "",
              "## Per-case results", "", "| Case | Original conclusion / agreement / grounded | Structural conclusion / agreement / grounded |",
              "|---|---|---|"]
    review = ["# Paired case review", "", "AI labels are provisional. Check source passages in each arm/contexts.jsonl.", ""]
    for key, pair in original.pairs.items():
        values = [" / ".join("not available" if arm.scores[key][metric] is None else
                             ("yes" if arm.scores[key][metric] else "no") for metric in METRICS)
                  for arm in (original, structural)]
        lines.append(f"| {key} | {values[0]} | {values[1]} |")
        review += [f"## {key}", "", "### Question", "", pair["question"], "", "### Reference", "", pair["reference"], ""]
        for name, arm in zip(ARMS, (original, structural)):
            row = arm.pairs[key]
            review += [f"### {name}", "", row["answer"], "",
                       "Gold document present in supplied evidence: " + str(row.get("gold_in_evidence", False)), ""]
            if key in arm.grades:
                review += ["```json", _json(arm.grades[key]).rstrip(), "```", ""]
    _write_reports(run, {"comparison.json": _json(summary), "by-group.json": _json(groups),
                         "comparison.md": "\n".join(lines) + "\n", "case-review.md": "\n".join(review) + "\n"})
    print(_json(summary))
    return summary


def write_calibration_report(run: Path) -> dict:
    arms = {arm: load_answer_run(run / arm, "v3", strict=True) for arm in ARMS}
    _paired_keys(arms["original"], arms["structural"])
    result = {"rubric": "answer-v3", "n": len(arms["original"].pairs), "groups": {},
              "v2_to_v3": {}, "interpretation": INTERPRETATION}
    lines = ["# Saved-answer v3 calibration comparison", "", INTERPRETATION, "",
             "Same saved answers and evidence; no retrieval or answer regeneration. V2 files remain unchanged.", "",
             "| Group | Metric | Paired resolved | Original | Structural | Wins | Losses |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group, keys in _groups(arms["original"]).items():
        result["groups"][group] = {}
        for metric, item in _metric_summary(arms["original"], arms["structural"], keys).items():
            n = item.get("paired_resolved", len(keys))
            summary = {"n": len(keys), "paired_resolved": n, "original": item["original"],
                       "structural": item["structural"], "improved": len(item["improved"]),
                       "regressed": len(item["regressed"])}
            result["groups"][group][metric] = summary
            lines.append(f"| {group} | {metric} | {n}/{len(keys)} | {summary['original']} | {summary['structural']} | "
                         f"{summary['improved']} | {summary['regressed']} |")
    changes = []
    for arm in ARMS:
        comparison = compare_rubric_versions(load_answer_run(run / arm, "v2", strict=True), arms[arm])
        result["v2_to_v3"][arm] = comparison["transitions"]
        changes.extend({"arm": arm, **change} for change in comparison["changes"])
    result["changed_labels"] = changes
    lines += ["", f"{len(changes)} arm-level metric records differ from v2. Inspect comparison-v3.json for both judgments and v3 checklists.", "",
              "Literal quote and checklist validation detects attribution/label inconsistencies, not entailment or completeness of the model-created checklist. Human calibration is still required.", ""]
    _write_reports(run, {"comparison-v3.json": _json(result), "comparison-v3.md": "\n".join(lines)})
    print("\n".join(lines))
    return result
