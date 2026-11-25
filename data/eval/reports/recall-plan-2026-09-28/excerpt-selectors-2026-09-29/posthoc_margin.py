"""POST-HOC, in-sample, exploratory (not part of the pre-registered design): how does a keep-the-prefix-unless-the-best-window-clearly-wins
rule behave? The margin is the cross-encoder score the best window must exceed the first window by. Writes nothing.

Run from the worktree root:  PYTHONPATH=src <python> <this file>
"""
import sys, json
from pathlib import Path
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import evaluate_selectors as ev
import windows as wd
eb = ev.eb

key = eb.judge_key(eb.MODEL, think=True)
capture_rows = {r["case_id"]: r for r in eb.read_jsonl(wd.LIVE / "per-case.jsonl")}
scores = ev.load_scores()
MARGINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
for name in eb.DATASETS:
    pool_dir = wd.LIVE / "grading" / name
    pool_rows = eb.read_jsonl(pool_dir / "pool.jsonl")
    pool = {(r["case_id"], r["chunk_id"]): r for r in pool_rows}
    judgments = eb.JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
    stats = {m: {"gain": 0, "lost": 0, "pairs_gain": 0, "pairs_lost": 0} for m in MARGINS}
    deviations = {m: 0 for m in MARGINS}
    docs_seen = 0
    for run in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl"):
        cid = run["case_id"]
        case = eb.make_case_views(cid, run["ranked"][:10], pool, judgments, capture_rows[cid])
        if not case["quotes"]:
            continue
        eligible = {q["pair"] for q in case["quotes"]}
        scored = scores[cid]
        first5 = list(zip(case["docs"], scored["docs"]))[:5]
        base = set()
        for doc, _ in first5:
            base |= eb.pair_set_for_window(doc["occurrences"], 0, min(1200, len(doc["joined"]))) & eligible
        for m in MARGINS:
            vis = set()
            for doc, sd in first5:
                length = len(doc["joined"])
                ws = sd["windows"]
                best = ev.choose([w["score"] for w in ws])
                first = ws[0]["score"]   # the window that starts at character 0
                if length <= 1200 or ws[best]["score"] - first <= m:
                    s, e = 0, min(1200, length)
                else:
                    s, e = wd.display_span(ws[best]["start"], ws[best]["end"], length)
                    deviations[m] += 1
                vis |= eb.pair_set_for_window(doc["occurrences"], s, e) & eligible
            st = stats[m]
            st["gain"] += int(bool(vis) and not base)
            st["lost"] += int(bool(base) and not vis)
            st["pairs_gain"] += len(vis - base); st["pairs_lost"] += len(base - vis)
        docs_seen += 1
    print(f"== {name} ({docs_seen} eligible cases)")
    for m in MARGINS:
        st = stats[m]
        print(f"  margin {m:4.2f}: docs deviating from prefix {deviations[m]:4d} | cases +{st['gain']}/-{st['lost']} net {st['gain']-st['lost']:+d} | pairs +{st['pairs_gain']}/-{st['pairs_lost']} net {st['pairs_gain']-st['pairs_lost']:+d}")
