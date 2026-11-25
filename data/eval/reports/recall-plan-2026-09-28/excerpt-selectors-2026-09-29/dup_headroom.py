"""OUTCOME-BLIND headroom check for a duplicate-aware document selector (not built): how many of the missing questions (no visible
verified grade-2 quote today, own-document credit) have a near-duplicate pair among the five shown documents, measured on the 1,200
characters Flask shows? Reads existing captures, pools and judgments only.

Run from the worktree root:  PYTHONPATH=src <python> <this file>
"""
import sys, re, itertools
from pathlib import Path
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import windows as wd
eb = wd.eb

def shingles(text, k=5):
    t = re.sub(r"\s+", " ", text)
    return {t[i:i+k] for i in range(0, max(1, len(t) - k + 1))}
def jaccard(a, b): return len(a & b) / len(a | b) if a and b else 0.0

key = eb.judge_key(eb.MODEL, think=True)
capture_rows = {r["case_id"]: r for r in eb.read_jsonl(wd.LIVE / "per-case.jsonl")}
for name in eb.DATASETS:
    pool_dir = wd.LIVE / "grading" / name
    pool = {(r["case_id"], r["chunk_id"]): r for r in eb.read_jsonl(pool_dir / "pool.jsonl")}
    judgments = eb.JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
    sims_all, rows = [], []
    for run in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl"):
        cid = run["case_id"]
        case = eb.make_case_views(cid, run["ranked"][:10], pool, judgments, capture_rows[cid])
        docs = case["docs"]
        shown = docs[:5]
        sh = [shingles(d["joined"][:1200]) for d in shown]
        pair_sims = [jaccard(sh[i], sh[j]) for i, j in itertools.combinations(range(len(shown)), 2)]
        sims_all += pair_sims
        eligible = {q["pair"] for q in case["quotes"]}
        base = set()
        for d in shown:
            base |= eb.pair_set_for_window(d["occurrences"], 0, min(1200, len(d["joined"]))) & eligible
        # is there any document in ranks 6-10 whose displayed prefix shows a quote? (necessary for a swap to help)
        later = set()
        for d in docs[5:]:
            later |= eb.pair_set_for_window(d["occurrences"], 0, min(1200, len(d["joined"]))) & eligible
        rows.append({"cid": cid, "eligible": bool(eligible), "missing": bool(eligible) and not base,
                     "max_sim": max(pair_sims) if pair_sims else 0.0, "later_prefix_quote": bool(later)})
    print(f"== {name}: {len(rows)} questions; displayed-text pair similarity >=0.5: {sum(s>=0.5 for s in sims_all)} of {len(sims_all)} pairs")
    for tau in (0.3, 0.5, 0.7):
        touched = [r for r in rows if r["max_sim"] >= tau]
        missing = [r for r in rows if r["missing"]]
        miss_dup = [r for r in missing if r["max_sim"] >= tau]
        miss_dup_later = [r for r in miss_dup if r["later_prefix_quote"]]
        print(f"  tau {tau}: questions with a duplicate pair {len(touched):3d} | of {len(missing)} missing questions, with a duplicate pair {len(miss_dup):2d}, "
              f"and a rank 6-10 document that shows a quote {len(miss_dup_later):2d}")
    # for reference: missing questions where ANY rank 6-10 doc shows a quote (what best5_prefix oracle exploits)
    missing = [r for r in rows if r["missing"]]
    print(f"  (reference) missing questions with a quote in some rank 6-10 prefix: {sum(r['later_prefix_quote'] for r in missing)} of {len(missing)}")
