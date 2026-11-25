"""Read-only: how much of a chunk does each stage look at? chunk lengths, reranker token coverage, where judge quotes sit."""
import json, re, statistics, collections
BASE = "data/eval/reports/recall-plan-2026-09-28/live-flask-current-2026-09-29-rootvenv"
MODEL, PROMPT = "ollama:qwen3:8b", "7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c"
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ws = lambda s: re.sub(r"\s+", " ", s).strip()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", local_files_only=True)
lens, cover, offs, rel_offs = [], [], [], []
pairs = 0
for ds in ("ip-200-fanout", "crossdomain-60-fanout"):
    G = f"{BASE}/grading/{ds}"
    text = {(r["case_id"], r["chunk_id"]): r for r in jl(f"{G}/pool.jsonl")}
    latest = {}
    for r in jl(f"{G}/judgments.jsonl"):
        if r.get("judge_model") == MODEL and r.get("prompt_sha256") == PROMPT: latest[(r["case_id"], r["chunk_id"])] = r
    for run in jl(f"{G}/runs/flask_current_k100.jsonl"):
        cid = run["case_id"]; q = text[(cid, run["ranked"][0]["chunk_id"])]["question"]
        qtok = len(tok(q)["input_ids"])
        for h in run["ranked"][:10]:
            row = text[(cid, h["chunk_id"])]; t = row["text"]; lens.append(len(t))
            j = latest[(cid, h["chunk_id"])]; d = j.get("details") or {}
            if j["grade"] == 2 and d.get("quote_verified") and d.get("evidence_quote"):
                nt, nq = ws(t), ws(d["evidence_quote"]).strip("'\"“”‘’…. ")
                p = nt.find(nq)
                if p >= 0: offs.append(p); rel_offs.append(p / max(1, len(nt)))
            pairs += 1
# reranker coverage: sentence-transformers CrossEncoder pads/truncates the (query, passage) pair to max_length=512
sample = []
for ds in ("ip-200-fanout",):
    for r in jl(f"{BASE}/grading/{ds}/pool.jsonl")[:400]:
        sample.append(r["text"])
for t in sample[:150]:
    ids = tok(t, add_special_tokens=False)["input_ids"]
    cover.append((len(t), len(ids)))
cpt = statistics.median(c / max(1, n) for c, n in cover)
budget_tokens = 512 - 45   # query ~30-40 tokens + 4 special tokens
print(f"top-10 chunks examined: {pairs} | chunk length chars: median {statistics.median(lens):.0f}, p10 {sorted(lens)[len(lens)//10]}, p90 {sorted(lens)[9*len(lens)//10]}, share > 1200 chars {sum(l > 1200 for l in lens)/len(lens):.0%}")
print(f"Korean chunk text: median {cpt:.2f} chars per reranker token -> the first ~{budget_tokens} tokens cover about {cpt*budget_tokens:.0f} chars of each chunk")
print(f"verified grade-2 quotes located in their chunk: {len(offs)} | quote start offset (normalised chars): median {statistics.median(offs):.0f}, p75 {sorted(offs)[3*len(offs)//4]}, p90 {sorted(offs)[9*len(offs)//10]}")
for lim in (1000, 1200, 2000):
    print(f"  quotes starting beyond char {lim}: {sum(o > lim for o in offs)/len(offs):.1%}")
