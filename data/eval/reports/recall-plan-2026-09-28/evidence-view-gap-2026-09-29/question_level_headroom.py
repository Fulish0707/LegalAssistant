"""Read-only: for the current served run, why does each question lack a visible verified grade-2 quote?"""
import json, re, collections
BASE = "data/eval/reports/recall-plan-2026-09-28/live-flask-current-2026-09-29-rootvenv"
MODEL, PROMPT = "ollama:qwen3:8b", "7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c"
ws = lambda s: re.sub(r"\s+", " ", s).strip()
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
percase = {r["case_id"]: r for r in jl(f"{BASE}/per-case.jsonl")}
for ds in ("ip-200-fanout", "crossdomain-60-fanout"):
    G = f"{BASE}/grading/{ds}"
    latest = {}
    for r in jl(f"{G}/judgments.jsonl"):
        if r.get("judge_model") == MODEL and r.get("prompt_sha256") == PROMPT: latest[(r["case_id"], r["chunk_id"])] = r
    runs = {r["case_id"]: r["ranked"][:10] for r in jl(f"{G}/runs/flask_current_k100.jsonl")}
    c = collections.Counter(); first_g2_rank = collections.Counter()
    for cid, ranked in runs.items():
        ev = percase[cid]["evidence"]; shown = {d["document_id"] for d in ev}; alltxt = ws(" ".join(d["text"] for d in ev))
        g = [(i + 1, h, (latest[(cid, h["chunk_id"])]["grade"]), latest[(cid, h["chunk_id"])].get("details") or {}) for i, h in enumerate(ranked)]
        g1 = [x for x in g if x[2] >= 1]; g2 = [x for x in g if x[2] == 2]
        if not g1: c["A no relevant chunk in top 10 (retrieval/ranking)"] += 1; continue
        if not g2: c["B relevant but never grade 2 (ranking/weak evidence)"] += 1; continue
        first_g2_rank[min(x[0] for x in g2)] += 1
        vis = any(x[3].get("quote_verified") and x[3].get("evidence_quote") and ws(x[3]["evidence_quote"]) in alltxt for x in g2)
        if vis:
            c["OK verified grade-2 quote visible"] += 1; continue
        in_shown = [x for x in g2 if x[1]["document_id"] in shown]
        if in_shown: c["C grade-2 chunk in a SHOWN document, quote cut (excerpt placement)"] += 1
        else: c["D every grade-2 chunk is in a document OUTSIDE the first five (document selection)"] += 1
    n = sum(c.values())
    print(f"\n=== {ds}: {n} questions (current served config) ===")
    for k in sorted(c): print(f"  {c[k]:4d} ({c[k]/n:5.1%})  {k}")
    late = sum(v for r, v in first_g2_rank.items() if r > 5); tot = sum(first_g2_rank.values())
    print(f"  of {tot} questions with a grade-2 chunk in the top 10, the FIRST one sits at rank 6-10 for {late} ({late/tot:.1%}); at rank 1 for {first_g2_rank[1]} ({first_g2_rank[1]/tot:.1%})")
