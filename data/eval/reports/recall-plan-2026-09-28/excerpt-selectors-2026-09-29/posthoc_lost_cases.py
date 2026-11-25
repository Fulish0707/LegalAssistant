"""POST-HOC, exploratory: print four IP questions where the cross-encoder window (ce_window_top5) lost a quote the prefix showed.

Run from the worktree root, after evaluate_selectors.py:  PYTHONPATH=src <python> <this file>
"""
import sys, json
from pathlib import Path
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import evaluate_selectors as ev
import windows as wd
eb = ev.eb
detail = json.loads((D / "selector-cases.json").read_text())
key = eb.judge_key(eb.MODEL, think=True)
capture_rows = {r["case_id"]: r for r in eb.read_jsonl(wd.LIVE / "per-case.jsonl")}
scores = ev.load_scores()
name = "ip-200-fanout"
lost = detail[name]["ce_window_top5"]["lost"][:4]
pool_dir = wd.LIVE / "grading" / name
pool = {(r["case_id"], r["chunk_id"]): r for r in eb.read_jsonl(pool_dir / "pool.jsonl")}
judgments = eb.JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
for cid in lost:
    run = next(r for r in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl") if r["case_id"] == cid)
    case = eb.make_case_views(cid, run["ranked"][:10], pool, judgments, capture_rows[cid])
    sc = scores[cid]
    print("=" * 100); print("Q:", sc["query"])
    for (doc, sd) in list(zip(case["docs"], sc["docs"]))[:5]:
        qs = [q for q in case["quotes"] if q["document_id"] == doc["document_id"]]
        if not qs: continue
        ws = sd["windows"]; best = ev.choose([w["score"] for w in ws])
        span = wd.display_span(ws[best]["start"], ws[best]["end"], len(doc["joined"]))
        print(f"-- doc {doc['document_id'][:40]}  len={len(doc['joined'])}  windows={len(ws)}  first-window score={ws[0]['score']:.3f}  best={ws[best]['score']:.3f} (window #{best})  shown={span}")
        for q in qs:
            occ = doc["occurrences"].get(q["pair"]) or []
            print(f"   quote@{occ[:1]} in prefix={'yes' if any(r<=1200 for l,r in occ) else 'no'}: {q['quote'][:140]!r}")
        print("   PREFIX :", doc["joined"][:160].replace("\n", " "))
        print("   CE-SHOWN:", doc["joined"][span[0]:span[0]+160].replace("\n", " "))
