"""Descriptive: in the current Flask top 10, how relevant is a slot by the corpus it comes from? Reads existing judgments only.

Run from the worktree root:  PYTHONPATH=src <python> <this file>
"""
import sys, statistics
from pathlib import Path
from collections import defaultdict
D = Path(__file__).resolve().parents[1] / "excerpt-selectors-2026-09-29"
sys.path.insert(0, str(D))
import windows as wd
eb = wd.eb
from legal_rag.cli import settings_from_options
from legal_rag.evaluation.crossdomain_target import corpus_document_ids
from legal_rag.factory import corpus_settings

settings = settings_from_options(profile="local", root=Path("/home/layke/Documents/MyLLMAgent"))
corpora = {v: corpus_document_ids(corpus_settings(settings, v).processed_corpus_dir())
           for v in ("ip-v1", "criminal-v1", "civil-v1", "administrative-v1")}
owner = {}
for v, ids in corpora.items():
    for i in ids:
        owner.setdefault(i, []).append(v)
print({v: len(ids) for v, ids in corpora.items()}, "ambiguous ids:", sum(len(o) > 1 for o in owner.values()))
key = eb.judge_key(eb.MODEL, think=True)
for name in eb.DATASETS:
    pool_dir = wd.LIVE / "grading" / name
    judgments = eb.JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
    stats = defaultdict(lambda: [0, 0, 0])  # slots, grade>=1, grade2
    unknown = 0
    per_case_share = []
    for run in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl"):
        cid = run["case_id"]; ip_slots = 0
        for hit in run["ranked"][:10]:
            o = owner.get(hit["document_id"])
            c = o[0] if o else "unknown"
            unknown += c == "unknown"
            g = judgments[(cid, hit["chunk_id"])]["grade"]
            s = stats[c]; s[0] += 1; s[1] += g >= 1; s[2] += g == 2
            ip_slots += c == "ip-v1"
        per_case_share.append(ip_slots / 10)
    print(f"== {name} (current config), unattributed slots: {unknown}")
    tot = sum(s[0] for s in stats.values())
    for c, (n, g1, g2) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:18s} slots {n:5d} ({n/tot:5.1%})  grade>=1 {g1/n:5.1%}  grade2 {g2/n:5.1%}")
    print(f"  mean ip-v1 share of top 10: {statistics.mean(per_case_share):.2f}")
