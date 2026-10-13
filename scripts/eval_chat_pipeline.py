#!/usr/bin/env python
"""Measure the /v1/chat components on gold-document questions.

`rewrite` asks whether query analysis and rewriting change retrieval: the question alone
(R0), rule rewriting (R1), R1 plus the analyzer's hard filter (R1F) and LLM rewriting (R2).
`verify` runs whole turns and records what the verification policy does to each draft,
with citation_guard as a second checker on the text the user sees. `report` prints both.
Results are resumable JSONL; both runs drive the real components in handle()'s order.

Two live runs of the same questions retrieve different evidence for about four in ten, so to
compare a prompt, verifier or generator change, `freeze` the analysis and retrieved chunks once
and `verify --frozen` each arm on them. `overlap` compares two judged runs without the grader.

Example (one heavy process at a time; unload Ollama models first):
    uv run python scripts/eval_chat_pipeline.py rewrite --out data/eval/reports/chat_pipeline/formal
    uv run python scripts/eval_chat_pipeline.py verify --out data/eval/reports/chat_pipeline/formal
    uv run python scripts/eval_chat_pipeline.py report --out data/eval/reports/chat_pipeline/formal
    uv run python scripts/eval_chat_pipeline.py freeze --out data/eval/reports/chat_pipeline/ab
    uv run python scripts/eval_chat_pipeline.py verify --frozen data/eval/reports/chat_pipeline/ab/frozen.jsonl --out data/eval/reports/chat_pipeline/ab/arm-a
    uv run python scripts/eval_chat_pipeline.py overlap --base data/eval/reports/chat_pipeline/ab/arm-a --new data/eval/reports/chat_pipeline/ab/arm-b
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
import time
import uuid
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp"))  # citation_guard lives with the web app

app = typer.Typer(add_completion=False)
DOMAINS = ("administrative", "civil", "criminal")  # ip sits at the exact-gold floor (~1%)


def _sample(per_domain: int, seed: int, cross: int | None) -> list[tuple[str, dict]]:
    rows = []
    for domain in DOMAINS:
        items = [json.loads(line) for line in open(ROOT / f"data/eval/{domain}-pilot-200.jsonl")]
        rows += [(f"{domain}-pilot-200", item) for item in random.Random(seed).sample(items, per_domain)]
    cross_items = [json.loads(line) for line in open(ROOT / "data/eval/targets/crossdomain-60/target.jsonl")]
    picked = cross_items if cross is None else random.Random(seed).sample(cross_items, cross)
    return rows + [("crossdomain-60", item) for item in picked]


def _questions(path: Path | None, per_domain: int, seed: int, cross: int | None) -> list[tuple[str, dict]]:
    """A custom JSONL ({id, query, relevant_document_ids, dataset?}) or the pilot sample."""
    if path is None:
        return _sample(per_domain, seed, cross)
    return [(item.get("dataset", path.stem), item) for item in map(json.loads, open(path))]


def _done(path: Path) -> set[str]:
    return {json.loads(line)["key"] for line in open(path)} if path.exists() else set()


def _set_aside_errors(path: Path) -> None:
    """Error rows (a rate limit, a dropped connection) are not results: move them out so a rerun redoes them."""
    if not path.exists():
        return
    rows = [json.loads(line) for line in open(path)]
    errors = [row for row in rows if row.get("kind") == "error"]
    if errors:
        with open(str(path) + ".errors", "a") as fh:
            fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in errors)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row.get("kind") != "error"))


def _docs(hits) -> list[str]:
    out: list[str] = []
    for hit in hits:
        if hit.chunk.document_id not in out:
            out.append(hit.chunk.document_id)
    return out


def _orchestrator(overrides: str | None = None, *, frozen: bool = False):
    """The local profile without clarification; `overrides` is JSON merged over it, e.g. an evidence arm.

    `frozen` swaps the retriever and analyzer for ones that replay a recorded turn, so no index is loaded."""
    from legal_rag.cli import settings_from_options
    from legal_rag.factory import build_orchestrator

    extra: dict = {"rag": {"allow_clarification": False}}
    for section, values in json.loads(overrides or "{}").items():
        extra.setdefault(section, {}).update(values)
    settings = settings_from_options(profile="local", extra=extra)
    if not frozen:
        return build_orchestrator(settings)
    from legal_rag.evaluation.answers.frozen import FrozenAnalyzer, FrozenRetriever

    orch = build_orchestrator(settings, retriever=FrozenRetriever())
    orch.analyzer = FrozenAnalyzer()
    return orch


@app.command()
def freeze(
    out: Path = typer.Option(..., help="Directory for frozen.jsonl."),
    questions: Path = typer.Option(None, exists=True, help="Custom question JSONL instead of the pilot sample."),
    per_domain: int = typer.Option(20), seed: int = typer.Option(11), cross: int = typer.Option(20),
    overrides: str = typer.Option(None, help="Settings JSON for the retrieval being frozen."),
) -> None:
    """Record each question's analysis and retrieved chunks, without generating an answer."""
    from legal_rag.evaluation.answers.frozen import capture_turn

    out.mkdir(parents=True, exist_ok=True)
    path = out / "frozen.jsonl"
    orch = _orchestrator(overrides)
    done = _done(path)
    with path.open("a") as fh:
        for dataset, ex in _questions(questions, per_domain, seed, cross):
            key = f"{dataset}:{ex['id']}"
            if key in done:
                continue
            turn = capture_turn(orch, ex["query"])
            fh.write(json.dumps({"key": key, "dataset": dataset, "gold": ex["relevant_document_ids"], **turn}, ensure_ascii=False) + "\n")
            fh.flush()


@app.command()
def rewrite(
    out: Path = typer.Option(..., help="Directory for rewrite.jsonl."),
    questions: Path = typer.Option(None, exists=True, help="Custom question JSONL instead of the pilot sample."),
    per_domain: int = typer.Option(50), seed: int = typer.Option(7),
) -> None:
    from legal_rag.rag.rewriter import QueryRewriter

    out.mkdir(parents=True, exist_ok=True)
    path = out / "rewrite.jsonl"
    orch = _orchestrator()
    llm_rewriter = QueryRewriter(orch.generator.model)
    done = _done(path)
    with path.open("a") as fh:
        for dataset, ex in _questions(questions, per_domain, seed, None):
            key = f"{dataset}:{ex['id']}"
            if key in done:
                continue
            q, started = ex["query"], time.perf_counter()
            analysis = orch.analyzer.analyze(q, [])
            rule = QueryRewriter(None).rewrite(q, analysis).search_queries
            llm = llm_rewriter.rewrite(q, analysis).search_queries
            filters = QueryRewriter.filters(analysis)
            r0 = _docs(orch.retriever.retrieve(q, 10, None))
            r1 = _docs(orch.retriever.retrieve_many(rule, 10, None, rerank_query=q))
            r1f = _docs(orch.retriever.retrieve_many(rule, 10, filters, rerank_query=q)) if filters else r1
            r2 = _docs(orch.retriever.retrieve_many(llm, 10, None, rerank_query=q)) if llm != rule else r1
            fh.write(json.dumps({
                "key": key, "dataset": dataset, "query": q, "gold": ex["relevant_document_ids"],
                "needs_more_info": analysis.needs_more_info,
                "filter": filters.citation_kind.value if filters and filters.citation_kind else None,
                "rule_queries": rule, "llm_queries": llm, "R0": r0, "R1": r1, "R1F": r1f, "R2": r2,
                "seconds": round(time.perf_counter() - started, 1),
            }, ensure_ascii=False) + "\n")
            fh.flush()


@app.command()
def verify(
    out: Path = typer.Option(..., help="Directory for verify.jsonl."),
    questions: Path = typer.Option(None, exists=True, help="Custom question JSONL instead of the pilot sample."),
    per_domain: int = typer.Option(20), seed: int = typer.Option(11), cross: int = typer.Option(20),
    overrides: str = typer.Option(None, help='Settings JSON, e.g. \'{"retrieval": {"k_final": 8}}\'.'),
    frozen: Path = typer.Option(None, exists=True, dir_okay=False, help="A frozen.jsonl to replay instead of analyzing and retrieving."),
    answer_prompt: Path = typer.Option(None, exists=True, dir_okay=False,
                                       help="Use this file as the answer system prompt for this arm, leaving the served prompt untouched."),
) -> None:
    import legal_rag.rag.orchestrator as orchestrator_module
    from citation_guard import unsupported_citations
    from legal_rag.verification.citations import answer_text
    from legal_rag.verification.policy import STRIP_NOTE

    out.mkdir(parents=True, exist_ok=True)
    path = out / "verify.jsonl"
    _set_aside_errors(path)
    orch = _orchestrator(overrides, frozen=frozen is not None)
    if answer_prompt is not None:
        import legal_rag.rag.generator as generator_module

        prompt_text, load_served = answer_prompt.read_text(encoding="utf-8"), generator_module.load_prompt
        generator_module.load_prompt = lambda name: prompt_text if name == "answer_system" else load_served(name)
    served = (ROOT / "src/legal_rag/llm/prompts/answer_system.md").read_bytes()
    (out / "arm.json").write_text(json.dumps({
        "model": getattr(orch.generator.model, "name", None), "frozen": str(frozen) if frozen else None, "overrides": overrides,
        "answer_prompt": str(answer_prompt) if answer_prompt else "served",
        "answer_prompt_sha256": hashlib.sha256(answer_prompt.read_bytes() if answer_prompt else served).hexdigest(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    turns: dict = {}
    if frozen is not None:
        from legal_rag.evaluation.answers.frozen import load_frozen

        turns = load_frozen(frozen)
    log: dict = {}

    def evidence(bundle) -> list[str]:
        return [" ".join(map(str, item.header.values())) + " " + item.text for item in bundle.items]

    def issues(report) -> list[str]:
        return [f"{issue.kind}:{issue.detail}" for issue in report.issues]

    original_backfill = orchestrator_module.backfill_citations
    original_verify, original_decide = orch.verifier.verify, orch.policy.decide
    frozen_analyzer = orch.analyzer  # the spy below replaces .analyze; a frozen analyzer still needs .load
    original_analyze = orch.analyzer.analyze

    def analyze_spy(message, history):
        analysis = original_analyze(message, history)
        log["user_refs"] = [*analysis.statute_refs, *analysis.case_refs]
        return analysis

    def backfill_spy(answer, bundle):
        raw = original_verify(answer, bundle, user_refs=log["user_refs"])
        log["attempts"].append({"raw_ok": raw.ok, "raw_issues": issues(raw),
                                "raw_guard": unsupported_citations(answer_text(answer), evidence(bundle) + [log["q"]])})
        return original_backfill(answer, bundle)

    def verify_spy(answer, bundle, **exemptions):
        report = original_verify(answer, bundle, **exemptions)
        log["attempts"][-1].update({"ok": report.ok, "issues": issues(report)})
        return report

    def decide_spy(report, attempt):
        action = original_decide(report, attempt)
        log["attempts"][-1]["action"] = action.value
        return action

    orchestrator_module.backfill_citations = backfill_spy
    orch.verifier.verify, orch.policy.decide, orch.analyzer.analyze = verify_spy, decide_spy, analyze_spy
    done = _done(path)
    with path.open("a") as fh:
        for dataset, ex in _questions(questions, per_domain, seed, cross):
            key = f"{dataset}:{ex['id']}"
            if key in done:
                continue
            log.clear()
            log.update({"attempts": [], "q": ex["query"], "user_refs": []})
            if frozen is not None:
                orch.retriever.load(turns[key])
                frozen_analyzer.load(turns[key])
            started = time.perf_counter()
            try:
                response = orch.handle(ex["query"], conversation_id=uuid.uuid4().hex)
            except Exception as error:  # e.g. GenerationError on truncated JSON: /v1/chat would 500
                fh.write(json.dumps({"key": key, "dataset": dataset, "kind": "error", "attempts": log["attempts"],
                                     "error": f"{type(error).__name__}: {str(error)[:300]}",
                                     "seconds": round(time.perf_counter() - started, 1)}, ensure_ascii=False) + "\n")
                fh.flush()
                continue
            record = {"key": key, "dataset": dataset, "kind": response.kind, "attempts": log["attempts"],
                      "regenerations": response.trace.regenerations if response.trace else None,
                      "final_ok": response.verification.ok if response.verification else None,
                      "final_issues": issues(response.verification) if response.verification else None,
                      "seconds": round(time.perf_counter() - started, 1)}
            if response.evidence is not None:
                record["evidence"] = evidence(response.evidence)
            if response.answer is not None:
                # The strip note names the citations it removed; the guard would count those as cited.
                text = answer_text(response.answer).split(STRIP_NOTE)[0]
                record["final_guard"] = unsupported_citations(text, evidence(response.evidence) + [ex["query"]])
                record["final_text"] = text[:1500]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()


@app.command()
def coverage(
    rewrite_run: Path = typer.Option(..., exists=True, dir_okay=False, help="A rewrite.jsonl whose rule_queries to reuse."),
    out: Path = typer.Option(..., help="Directory for coverage.jsonl."),
    depth: int = typer.Option(20),
) -> None:
    """Is the gold document among the evidence /v1/chat would build, at several evidence sizes?

    Replays the stored rule-rewrite queries through retrieve_many, as handle() does, so no LLM
    call is needed. One ranked list of `depth` chunks serves every size: the reranker only cuts."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "coverage.jsonl"
    orch = _orchestrator()
    queries = {f"{name}:{item['id']}": item["query"] for name, item in _sample(50, 7, None)}
    done = _done(path)
    with path.open("a") as fh:
        for row in map(json.loads, open(rewrite_run)):
            if row["key"] in done:
                continue
            query = row.get("query") or queries[row["key"]]
            hits = orch.retriever.retrieve_many(row["rule_queries"], depth, None, rerank_query=query)
            fh.write(json.dumps({"key": row["key"], "dataset": row["dataset"], "gold": row["gold"],
                                 "chunk_documents": [hit.chunk.document_id for hit in hits],
                                 "chunk_chars": [len(hit.chunk.text) for hit in hits]}, ensure_ascii=False) + "\n")
            fh.flush()


@app.command()
def overlap(
    base: Path = typer.Option(..., exists=True, file_okay=False, help="Run directory with answer_pairs.jsonl."),
    new: Path = typer.Option(..., exists=True, file_okay=False, help="Run directory with answer_pairs.jsonl."),
    chars: int = typer.Option(700, help="Compare the first N characters of every answer; recall grows with length."),
) -> None:
    """Compare two runs' answers with their references without the grader (overlap, not correctness)."""
    from legal_rag.evaluation.answers.reference_overlap import bigram_recall, fact_recall, paired_difference

    runs = [{row["key"]: row for row in map(json.loads, open(run / "answer_pairs.jsonl"))} for run in (base, new)]
    keys = sorted(set(runs[0]) & set(runs[1]))
    scores = []
    for run in runs:
        recall = [bigram_recall(run[key]["answer"][:chars], run[key]["reference"]) for key in keys]
        facts = [fact_recall(run[key]["answer"][:chars], run[key]["reference"]) for key in keys]
        scores.append((recall, facts))
    typer.echo(f"{len(keys)} shared questions, first {chars} characters of each answer")
    for name, (recall, facts) in zip(("base", "new"), scores):
        present = [value for value in facts if value is not None]
        typer.echo(f"  {name}: bigram recall {sum(recall) / len(recall):.3f} | article and quantity recall "
                   f"{sum(present) / len(present):.3f} (n={len(present)})")
    mean, (low, high) = paired_difference(scores[0][0], scores[1][0])
    typer.echo(f"  new - base, bigram recall: {mean:+.3f} [{low:+.3f}, {high:+.3f}]")
    both = [(a, b) for a, b in zip(scores[0][1], scores[1][1]) if a is not None and b is not None]
    mean, (low, high) = paired_difference([a for a, _ in both], [b for _, b in both])
    typer.echo(f"  new - base, article and quantity recall: {mean:+.3f} [{low:+.3f}, {high:+.3f}]")


def _rank(top: list[str], gold: list[str]) -> int | None:
    return next((i for i, doc in enumerate(top[:10], 1) if doc in gold), None)


def _bootstrap(pairs: list[tuple[float, float]], iters: int = 2000) -> tuple[float, float]:
    rnd, diffs = random.Random(0), []
    for _ in range(iters):
        draw = [rnd.choice(pairs) for _ in pairs]
        diffs.append(sum(b - a for a, b in draw) / len(draw))
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


@app.command()
def report(out: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    rewrite_path, verify_path = out / "rewrite.jsonl", out / "verify.jsonl"
    if rewrite_path.exists():
        rows = [json.loads(line) for line in open(rewrite_path)]
        groups = collections.defaultdict(list)
        for row in rows:
            groups["ALL"].append(row)
            groups[row["dataset"]].append(row)
        typer.echo("== rewrite: gold document in top 10, paired against R0 (question only)")
        for condition in ("R1", "R1F", "R2"):
            typer.echo(f"-- {condition}")
            for name, members in groups.items():
                s10 = [(float(bool(_rank(m["R0"], m["gold"]))), float(bool(_rank(m[condition], m["gold"])))) for m in members]
                mrr = [tuple(1 / r if (r := _rank(m[c], m["gold"])) else 0.0 for c in ("R0", condition)) for m in members]
                (lo, hi), (mlo, mhi) = _bootstrap(s10), _bootstrap(mrr)
                a, b = sum(x for x, _ in s10) / len(s10), sum(y for _, y in s10) / len(s10)
                am, bm = sum(x for x, _ in mrr) / len(mrr), sum(y for _, y in mrr) / len(mrr)
                typer.echo(f"   {name:26} n={len(members):3} S@10 {a:.3f}->{b:.3f} [{lo:+.3f},{hi:+.3f}]  "
                           f"MRR {am:.3f}->{bm:.3f} [{mlo:+.3f},{mhi:+.3f}]")
        fired = [row for row in rows if row["filter"]]
        typer.echo(f"filter fired {len(fired)}/{len(rows)} {dict(collections.Counter(r['filter'] for r in fired))}; "
                   f"clarify {sum(r['needs_more_info'] for r in rows)}/{len(rows)}; "
                   f"LLM rewrite changed queries {sum(r['llm_queries'] != r['rule_queries'] for r in rows)}/{len(rows)}")
    if verify_path.exists():
        rows = [json.loads(line) for line in open(verify_path)]
        first = [row["attempts"][0] for row in rows if row["attempts"]]
        answered = [row for row in rows if "final_guard" in row]
        typer.echo("== verify")
        typer.echo(f"turns {len(rows)} | final kind {dict(collections.Counter(r['kind'] for r in rows))} | "
                   f"regenerated {sum(1 for r in rows if (r.get('regenerations') or 0) > 0)}")
        typer.echo(f"first draft flagged by verifier {sum(1 for a in first if not a['raw_ok'])}/{len(first)}; "
                   f"issue kinds {dict(collections.Counter(i.split(':')[0] for a in first for i in a['raw_issues']))}")
        typer.echo(f"citation_guard: first draft {sum(1 for a in first if a['raw_guard'])}/{len(first)}, "
                   f"final shown {sum(1 for r in answered if r['final_guard'])}/{len(answered)}")
        typer.echo(f"mean seconds per turn {sum(r['seconds'] for r in rows) / len(rows):.1f}")


if __name__ == "__main__":
    app()
