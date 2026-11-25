#!/usr/bin/env python3
"""Evaluate label-free excerpt selectors against the evidence-budget oracles (design.md is the contract).

Reads window-scores.jsonl (score_windows.py) and the same captured top 10, pool, judgments and quote matching as
``priority1-evidence-budget-report.md``, whose functions are imported. Labels are used only to count which verified
quotes a selector's excerpts show; no selector reads them. Writes selector-analysis.json, selector-cases.json and
selector-report.md next to this file.

Run from the worktree root, after score_windows.py:
    PYTHONPATH=src <python> <this file>
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import windows as wd  # noqa: E402

eb = wd.eb
from legal_rag.retrieval.tokenizer import RegexTokenizer  # noqa: E402

VARIANTS = ("prefix", "ce_window_top5", "lex_window_top5", "ce_docs_prefix", "ce_docs_window", "lex_docs_window")
LABELS = {
    "prefix": "Current top 5 + prefix",
    "ce_window_top5": "Same top 5 + cross-encoder window",
    "lex_window_top5": "Same top 5 + lexical window",
    "ce_docs_prefix": "Top 5 by cross-encoder window + prefix",
    "ce_docs_window": "Top 5 by cross-encoder window + its window",
    "lex_docs_window": "Top 5 by lexical window + its window",
}
ORACLE = {  # the oracle row each selector is measured against
    "ce_window_top5": "current_top5_char_window_oracle", "lex_window_top5": "current_top5_char_window_oracle",
    "ce_docs_prefix": "best5_prefix_oracle",
    "ce_docs_window": "best5_char_window_oracle", "lex_docs_window": "best5_char_window_oracle",
}
ORACLE_LABELS = {
    "current_top5_char_window_oracle": "Oracle: same top 5 + character window",
    "best5_prefix_oracle": "Oracle: best 5 + prefix",
    "best5_char_window_oracle": "Oracle: best 5 + character window",
}
TOKENIZER = RegexTokenizer()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_p(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_scores() -> dict[str, dict]:
    return {row["case_id"]: row for row in eb.read_jsonl(BASE / "window-scores.jsonl")}


def idf_table(pool_rows: list[dict]) -> tuple[dict[str, float], int]:
    """Chunk-level document frequency over the dataset's pool (unique chunks), BM25-style idf."""
    seen, frequency = set(), Counter()
    for row in pool_rows:
        if row["chunk_id"] in seen:
            continue
        seen.add(row["chunk_id"])
        frequency.update(set(TOKENIZER.tokenize(row.get("text") or "")))
    total = len(seen)
    return {term: math.log((total - df + 0.5) / (df + 0.5) + 1.0) for term, df in frequency.items()}, total


def lexical_scores(query: str, doc: dict, spans: list[tuple[int, int]], idf: dict[str, float]) -> list[float]:
    query_terms = set(TOKENIZER.tokenize(query))
    return [sum(idf.get(term, 0.0) for term in query_terms & set(TOKENIZER.tokenize(doc["joined"][s:e])))
            for s, e in spans]


def choose(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def shown_documents(case: dict, scored: dict, idf: dict[str, float]) -> dict[str, list[tuple[dict, tuple[int, int]]]]:
    """For every variant, the (document, displayed span) list Flask would hand the model."""
    docs = case["docs"]
    if [d["document_id"] for d in docs] != [d["document_id"] for d in scored["docs"]]:
        raise ValueError(f"document order differs from the scored windows for {case['case_id']}")
    per_doc = {}
    for doc, scored_doc in zip(docs, scored["docs"]):
        spans = [(w["start"], w["end"]) for w in scored_doc["windows"]]
        if len(doc["joined"]) != scored_doc["joined_chars"] or not spans:
            raise ValueError(f"scored windows do not match the document text for {case['case_id']}")
        ce = [w["score"] for w in scored_doc["windows"]]
        lex = lexical_scores(scored["query"], doc, spans, idf)
        views = {}
        for name, values in (("ce", ce), ("lex", lex)):
            best = choose(values)
            views[name] = {"score": values[best], "span": wd.display_span(*spans[best], len(doc["joined"]))}
        per_doc[doc["document_id"]] = views
    prefix = lambda doc: (0, min(wd.CHAR_LIMIT, len(doc["joined"])))
    first5 = docs[:wd.MAX_DOCUMENTS]

    def top5(kind: str) -> list[dict]:
        ranked = sorted(docs, key=lambda d: (-per_doc[d["document_id"]][kind]["score"], d["rank"]))
        return ranked[:wd.MAX_DOCUMENTS]

    return {
        "prefix": [(d, prefix(d)) for d in first5],
        "ce_window_top5": [(d, per_doc[d["document_id"]]["ce"]["span"]) for d in first5],
        "lex_window_top5": [(d, per_doc[d["document_id"]]["lex"]["span"]) for d in first5],
        "ce_docs_prefix": [(d, prefix(d)) for d in top5("ce")],
        "ce_docs_window": [(d, per_doc[d["document_id"]]["ce"]["span"]) for d in top5("ce")],
        "lex_docs_window": [(d, per_doc[d["document_id"]]["lex"]["span"]) for d in top5("lex")],
    }


def window_hits(case: dict, scored: dict, idf: dict[str, float]) -> list[dict]:
    """Does the best-scoring window (before widening) contain each verified quote? Measures the score, not the display."""
    rows = []
    by_doc = {d["document_id"]: (d, s) for d, s in zip(case["docs"], scored["docs"])}
    for quote in case["quotes"]:
        doc, scored_doc = by_doc[quote["document_id"]]
        locations = doc["occurrences"].get(quote["pair"]) or []
        spans = [(w["start"], w["end"]) for w in scored_doc["windows"]]
        contains = [any(s <= left and right <= e for left, right in locations) for s, e in spans]
        if not any(contains):
            continue  # quote straddles a window edge, or spans two chunks; no window can be credited
        ce = [w["score"] for w in scored_doc["windows"]]
        lex = lexical_scores(scored["query"], doc, spans, idf)
        row = {"windows": len(spans), "containing": sum(contains)}
        for name, values in (("ce", ce), ("lex", lex)):
            order = sorted(range(len(spans)), key=lambda i: (-values[i], i))
            row[f"{name}_top1"] = contains[order[0]]
            row[f"{name}_top3"] = any(contains[i] for i in order[:3])
        row["first_window"] = contains[0]
        row["chance_top1"] = sum(contains) / len(spans)
        rows.append(row)
    return rows


def evaluate_dataset(name: str, key, capture_rows: dict, scores: dict) -> dict:
    pool_dir = wd.LIVE / "grading" / name
    pool_rows = eb.read_jsonl(pool_dir / "pool.jsonl")
    pool = {(r["case_id"], r["chunk_id"]): r for r in pool_rows}
    judgments = eb.JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
    idf, chunks = idf_table(pool_rows)
    cases, oracle_views, selector_views, hits = [], {}, {}, []
    for run in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl"):
        case = eb.make_case_views(run["case_id"], run["ranked"][:10], pool, judgments, capture_rows[run["case_id"]])
        case["dataset"] = name
        cases.append(case)
        oracle_views[case["case_id"]] = eb.variants_for_case(case)
        selector_views[case["case_id"]] = shown_documents(case, scores[case["case_id"]], idf)
        hits.extend(window_hits(case, scores[case["case_id"]], idf))

    oracle_summary = eb.summarize_dataset(name, cases, oracle_views)
    eligible = {c["case_id"]: {q["pair"] for q in c["quotes"]} for c in cases if c["quotes"]}
    result = {"eligible_cases": len(eligible), "verified_grade2_quotes": sum(len(p) for p in eligible.values()),
              "idf_chunks": chunks, "variants": {}, "window_hits": {}}
    visible = {v: {} for v in VARIANTS}
    density = {v: {} for v in VARIANTS}
    for case in cases:
        cid = case["case_id"]
        if cid not in eligible:
            continue
        for variant in VARIANTS:
            shown = selector_views[cid][variant]
            pairs = set()
            with_quote = 0
            for doc, (start, end) in shown:
                got = eb.pair_set_for_window(doc["occurrences"], start, end) & eligible[cid]
                pairs |= got
                with_quote += bool(got)
            visible[variant][cid] = pairs
            density[variant][cid] = with_quote
    base = visible["prefix"]
    cases_detail = {}
    for variant in VARIANTS:
        gained_cases = sorted(c for c in eligible if visible[variant][c] and not base[c])
        lost_cases = sorted(c for c in eligible if base[c] and not visible[variant][c])
        gained_pairs = sum(len(visible[variant][c] - base[c]) for c in eligible)
        lost_pairs = sum(len(base[c] - visible[variant][c]) for c in eligible)
        pairs = sum(len(visible[variant][c]) for c in eligible)
        case_count = sum(bool(visible[variant][c]) for c in eligible)
        row = {"visible_quote_pairs": pairs, "visible_cases": case_count,
               "cases_gained": len(gained_cases), "cases_lost": len(lost_cases),
               "net_cases": case_count - sum(bool(base[c]) for c in eligible),
               "pairs_gained": gained_pairs, "pairs_lost": lost_pairs, "net_pairs": gained_pairs - lost_pairs,
               "sign_test_p_cases": sign_p(len(gained_cases), len(lost_cases)),
               "mean_shown_documents_with_visible_quote": statistics.mean(density[variant][c] for c in eligible)}
        if variant in ORACLE:
            oracle = oracle_summary["variants"][ORACLE[variant]]
            base_row = oracle_summary["variants"]["current_top5_prefix"]
            gap_cases = oracle["visible_cases_with_quote"] - base_row["visible_cases_with_quote"]
            gap_pairs = oracle["visible_quote_pairs"] - base_row["visible_quote_pairs"]
            row.update({"oracle": ORACLE[variant], "oracle_gap_cases": gap_cases, "oracle_gap_pairs": gap_pairs,
                        "share_of_case_gap_closed": row["net_cases"] / gap_cases if gap_cases else None,
                        "share_of_pair_gap_closed": row["net_pairs"] / gap_pairs if gap_pairs else None})
        result["variants"][variant] = row
        cases_detail[variant] = {"gained": gained_cases, "lost": lost_cases}
    result["oracle_rows"] = {k: {f: v[f] for f in ("visible_quote_pairs", "visible_cases_with_quote",
                                                   "newly_visible_cases_vs_current", "pair_change_vs_current")}
                             for k, v in oracle_summary["variants"].items()}
    result["oracle_summary"] = oracle_summary
    if hits:
        n = len(hits)
        result["window_hits"] = {
            "quotes_located_in_a_window": n,
            "cross_encoder_top1": sum(h["ce_top1"] for h in hits) / n, "cross_encoder_top3": sum(h["ce_top3"] for h in hits) / n,
            "lexical_top1": sum(h["lex_top1"] for h in hits) / n, "lexical_top3": sum(h["lex_top3"] for h in hits) / n,
            "first_window": sum(h["first_window"] for h in hits) / n,
            "chance_top1": sum(h["chance_top1"] for h in hits) / n}
    return result, cases_detail


def gate(result: dict, name: str, stored: dict) -> None:
    """The harness must reproduce the evidence-budget report before any selector is read."""
    mine = result["oracle_summary"]["variants"]
    theirs = stored["datasets"][name]["variants"]
    for variant, row in theirs.items():
        for field in ("visible_quote_pairs", "visible_cases_with_quote", "newly_visible_cases_vs_current", "pair_change_vs_current"):
            if mine[variant][field] != row[field]:
                raise SystemExit(f"GATE FAILED {name} {variant} {field}: {mine[variant][field]} != {row[field]}")
    if result["variants"]["prefix"]["visible_quote_pairs"] != theirs["current_top5_prefix"]["visible_quote_pairs"]:
        raise SystemExit(f"GATE FAILED {name}: my prefix rendering differs from Flask's")


def verdicts(datasets: dict) -> dict:
    out = {}
    for variant, cases_rule in (("ce_window_top5", "a"), ("ce_docs_window", "c")):
        checks = {}
        for name, data in datasets.items():
            row = data["variants"][variant]
            need = math.ceil(row["oracle_gap_cases"] / 2)
            ok = row["net_cases"] >= need and row["cases_lost"] <= 1 and (row["net_pairs"] > 0 if cases_rule == "a" else True)
            checks[name] = {"net_cases": row["net_cases"], "needed": need, "cases_lost": row["cases_lost"],
                            "net_pairs": row["net_pairs"], "pass": ok}
        out[variant] = {"pass": all(c["pass"] for c in checks.values()), "datasets": checks}
    return out


def write_report(data: dict, path: Path) -> None:
    short = {"ip-200-fanout": "IP (200)", "crossdomain-60-fanout": "Mixed domains (60)"}
    lines = [
        "# Priority 1 follow-up: label-free excerpt selectors against the evidence-budget oracles", "",
        "- **Date:** 2026-09-29",
        "- **Result type:** Development-only offline counterfactual; retrieval, ranking and serving unchanged",
        "- **Design and pass bar:** `design.md`, written before any selector was scored", "",
        "## Quote survival", "",
        "Same top-10 evidence, same formatter and quote matching as `priority1-evidence-budget-report.md`. Every row shows at "
        "most five documents and 1,200 characters per document. Oracle rows are copied from that report. Gained and lost count "
        "questions with a visible verified grade-2 quote against the current prefix; the gap closed is net new questions over "
        "the oracle's new questions.", "",
        "| Dataset | Evidence view | Quote pairs | Cases with a visible quote | Cases gained / lost | Net cases | Share of oracle case gap closed | Pairs gained / lost | Shown documents holding a quote (of 5) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, dataset in data["datasets"].items():
        total_pairs, eligible = dataset["verified_grade2_quotes"], dataset["eligible_cases"]
        for variant in VARIANTS:
            row = dataset["variants"][variant]
            closed = row.get("share_of_case_gap_closed")
            lines.append(f"| {short[name]} | {LABELS[variant]} | {row['visible_quote_pairs']}/{total_pairs} "
                         f"| {row['visible_cases']}/{eligible} | +{row['cases_gained']} / -{row['cases_lost']} "
                         f"| {row['net_cases']:+d} | {'' if closed is None else format(closed, '.0%')} "
                         f"| +{row['pairs_gained']} / -{row['pairs_lost']} | {row['mean_shown_documents_with_visible_quote']:.2f} |")
        for oracle in ("current_top5_char_window_oracle", "best5_prefix_oracle", "best5_char_window_oracle"):
            row = dataset["oracle_rows"][oracle]
            lines.append(f"| {short[name]} | {ORACLE_LABELS[oracle]} | {row['visible_quote_pairs']}/{total_pairs} "
                         f"| {row['visible_cases_with_quote']}/{eligible} | | +{row['newly_visible_cases_vs_current']} | 100% "
                         f"| | |")
    lines += ["", "## Pass bar (fixed in design.md)", "",
              "| Selector | Dataset | Net new cases | Needed | Cases lost (max 1) | Net pairs | Result |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for variant, verdict in data["verdicts"].items():
        for name, check in verdict["datasets"].items():
            lines.append(f"| {LABELS[variant]} | {short[name]} | {check['net_cases']:+d} | {check['needed']} | {check['cases_lost']} "
                         f"| {check['net_pairs']:+d} | {'pass' if check['pass'] else 'miss'} |")
        lines.append(f"| **{LABELS[variant]}** | **both** | | | | | **{'PASS' if verdict['pass'] else 'MISS'}** |")
    lines += ["", "## Does the score find the quote?", "",
              "Share of verified quotes that sit inside the highest-scoring window of their document (before it is widened to "
              "1,200 characters), inside one of the three highest, and inside the first window. Chance is the average share of a "
              "document's windows that contain the quote.", "",
              "| Dataset | Quotes located | Cross-encoder top 1 | Cross-encoder top 3 | Lexical top 1 | Lexical top 3 | First window | Chance (top 1) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, dataset in data["datasets"].items():
        h = dataset["window_hits"]
        lines.append(f"| {short[name]} | {h['quotes_located_in_a_window']} | {h['cross_encoder_top1']:.1%} | {h['cross_encoder_top3']:.1%} "
                     f"| {h['lexical_top1']:.1%} | {h['lexical_top3']:.1%} | {h['first_window']:.1%} | {h['chance_top1']:.1%} |")
    cost = data["cost"]
    lines += ["", "## Serving cost", "",
              f"Measured on the RTX 4060 laptop GPU, fp16, batch 16: {cost['ms_per_pair']:.1f} ms per (query, window) pair. "
              f"Scoring all windows of the ten documents takes {cost['mean_pairs_all_docs']:.0f} pairs, about {cost['mean_seconds_all_docs']:.2f} s "
              f"per question (measured); scoring only the first five documents takes {cost['mean_pairs_top5_docs']:.0f} pairs, about "
              f"{cost['estimated_seconds_top5_docs']:.2f} s (estimated from the per-pair time). For scale, `k_fused` 100 against 50 cost about 0.7 s.",
              "", "## Provenance", "",
              f"Judge: `ollama:qwen3:8b`, prompt `{eb.PROMPT_SHA256}`, `think=true`; no new judgments. Independent human calibration "
              "was unavailable. Each pair has one Qwen-chosen quote, so a selector that shows a different passage that also answers "
              "the question scores as a miss. Input hashes are in `selector-analysis.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    key = eb.judge_key(eb.MODEL, think=True)
    stored = json.loads((wd.LIVE / "priority1-evidence-budget-analysis.json").read_text(encoding="utf-8"))
    capture_rows = {row["case_id"]: row for row in eb.read_jsonl(wd.LIVE / "per-case.jsonl")}
    scores = load_scores()
    datasets, detail = {}, {}
    for name in eb.DATASETS:
        datasets[name], detail[name] = evaluate_dataset(name, key, capture_rows, scores)
        gate(datasets[name], name, stored)
        print(f"gate passed for {name}: baseline and oracle rows reproduce the evidence-budget report", flush=True)
    run = json.loads((BASE / "score-run.json").read_text(encoding="utf-8"))
    ms_per_pair = run["timing"]["seconds_per_pair_ms"]
    top5_pairs = statistics.mean(sum(len(d["windows"]) for d in scores[cid]["docs"][:5]) for cid in scores)
    cost = {"ms_per_pair": ms_per_pair, "mean_pairs_all_docs": run["timing"]["mean_pairs"],
            "mean_seconds_all_docs": run["timing"]["mean_seconds"], "mean_pairs_top5_docs": top5_pairs,
            "estimated_seconds_top5_docs": top5_pairs * ms_per_pair / 1000}
    result = {"evaluation": "development_only_counterfactual", "design": "design.md",
              "retrieval_or_serving_changed": False, "datasets": {n: {k: v for k, v in d.items() if k != "oracle_summary"}
                                                                   for n, d in datasets.items()},
              "verdicts": verdicts(datasets), "cost": cost,
              "input_sha256": {"design.md": sha256(BASE / "design.md"), "windows.py": sha256(BASE / "windows.py"),
                               "score_windows.py": sha256(BASE / "score_windows.py"),
                               "window-scores.jsonl": sha256(BASE / "window-scores.jsonl"),
                               "score-run.json": sha256(BASE / "score-run.json"),
                               "priority1-evidence-budget-analysis.json": sha256(wd.LIVE / "priority1-evidence-budget-analysis.json"),
                               "analyze_priority1_evidence_budget.py": sha256(wd.LIVE / "analyze_priority1_evidence_budget.py")}}
    (BASE / "selector-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "selector-cases.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    result["analysis_script_sha256"] = sha256(Path(__file__))
    (BASE / "selector-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, BASE / "selector-report.md")
    for name, dataset in result["datasets"].items():
        print(f"\n== {name}: {dataset['eligible_cases']} eligible cases, {dataset['verified_grade2_quotes']} quotes")
        for variant in VARIANTS:
            r = dataset["variants"][variant]
            closed = r.get("share_of_case_gap_closed")
            print(f"  {variant:16s} pairs {r['visible_quote_pairs']:4d}  cases {r['visible_cases']:3d}  +{r['cases_gained']}/-{r['cases_lost']} "
                  f"net {r['net_cases']:+d}  gap closed {'' if closed is None else format(closed, '.0%'):>5s}  "
                  f"pairs +{r['pairs_gained']}/-{r['pairs_lost']}  docs-with-quote {r['mean_shown_documents_with_visible_quote']:.2f}")
        print("  window hits:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dataset["window_hits"].items()}))
    print("\nverdicts:", json.dumps({v: r["pass"] for v, r in result["verdicts"].items()}))
    print("cost:", json.dumps({k: round(v, 3) for k, v in cost.items()}))


if __name__ == "__main__":
    main()
