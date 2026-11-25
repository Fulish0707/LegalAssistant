"""Case-law evaluation: script metrics (A1, A2) and blind grading packets (B1, B2).
  python score_caselaw.py packets   -> packets.jsonl (+ _key.jsonl), shuffled, arm/model hidden
  python score_caselaw.py report    -> A1/A2 tables, and B1/B2 once labels/*.jsonl exist
"""
import collections
import json
import random
import sys
from pathlib import Path

RUNS = Path("/home/layke/Documents/MyLLMAgent/data/eval/action_policy_caselaw/runs")
D = Path(__file__).parent
NOTICE = "인용한 판례·결정례 중"
EXPECT = {"K1-unconfirmed-case": "answer", "K2-confirmed-finding": "answer",
          "K3-claim-only": "withheld", "K4-claim-and-finding": "answer"}


PREFIX = "v2-"  # runs after the bent-tail fix; the first runs stay as the pre-fix record


def rows():
    for run in sorted(RUNS.iterdir()):
        if not run.name.startswith(PREFIX):
            continue
        model, rep = run.name.removeprefix(PREFIX).rsplit("-rep", 1)
        for line in (run / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            yield model, rep, r


def cited_sources(resp):
    by_s = {e["source_id"]: e["chunk_id"] for e in resp["evidence"]["items"]}
    return {c["source_id"]: by_s.get(c["source_id"], "?") for c in (resp.get("answer") or {}).get("citations", [])}


def short(chunk_id):
    return {"#reasoning": "판단창", "#claims": "주장창", "#summary": "요지(미상)"}.get(
        chunk_id[chunk_id.find("#"):] if "#" in chunk_id else "", "법령" if "official" in chunk_id else chunk_id)


def report():
    print("| 모델 | 회 | 사례 | 갈래 | 결과 | 정책 행동 | 판례 역할 | 인용 | 고정 안내 | 신뢰도 | A1 | A2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    a1, a2 = collections.Counter(), collections.Counter()
    for model, rep, r in rows():
        resp = r["response"] or {}
        kind = resp.get("kind", r["status"])
        trace = resp.get("trace") or {}
        acts = "→".join(d["action"] for d in trace.get("post_retrieval", [])) or "-"
        roles = {}
        for d in trace.get("post_retrieval", []):
            roles.update(d.get("support_roles", {}))
        by_s = {e["source_id"]: short(e["chunk_id"]) for e in resp.get("evidence", {}).get("items", [])} if resp else {}
        roles_txt = ",".join(f"{by_s.get(s, s)}={v}" for s, v in roles.items()) or "-"
        cited = cited_sources(resp) if resp else {}
        cited_txt = ",".join(sorted({short(c) for c in cited.values()})) or "-"
        ans = resp.get("answer") or {}
        notice = ans.get("uncertainty", "").startswith(NOTICE)
        ok1 = ok2 = ""
        if r["arm"] == "candidate":
            ok1 = "✓" if kind == EXPECT[r["case_id"]] else "✗"
            a1[(model, ok1)] += 1
            if kind == "answer":
                unconfirmed = set(trace.get("unverified_case_sources", []))
                cases_cited = {s for s, c in cited.items() if "fictional" in c}
                confirmed = {s for s in cases_cited if roles.get(s) in ("finding", "legal_rule")}
                need = bool(cases_cited - confirmed)
                good = (notice == need) and (unconfirmed == (cases_cited - confirmed)) and not (need and ans.get("confidence") == "high")
                ok2 = "✓" if good else "✗"
                a2[(model, ok2)] += 1
        print(f"| {model} | {rep} | {r['case_id'][:3]} | {r['arm'][:4]} | {kind} | {acts} | {roles_txt} | {cited_txt} | "
              f"{'있음' if notice else '-'} | {ans.get('confidence', '-')} | {ok1} | {ok2} |")
    print("\nA1", dict(a1), "\nA2", dict(a2))
    labels = list((D / "labels").glob("*.jsonl")) if (D / "labels").exists() else []
    if labels:
        key = {x["pid"]: x for x in map(json.loads, (D / "_key.jsonl").read_text().splitlines())}
        grades = {}
        for f in labels:
            for x in map(json.loads, f.read_text().splitlines()):
                grades[x["pid"]] = x
        table = collections.Counter()
        for pid, g in grades.items():
            k = key[pid]
            table[(k["case_id"][:2], k["arm"], k["model"], "B1", g["outcome"])] += 1
            table[(k["case_id"][:2], k["arm"], k["model"], "B2", str(g["claim_as_holding"]).lower())] += 1
        for row, n in sorted(table.items()):
            print(row, n)


def packets():
    items, key = [], []
    for model, rep, r in rows():
        resp = r["response"] or {}
        ans = resp.get("answer")
        if not ans:
            continue
        body = {"summary": ans["summary"], "analysis": ans["analysis"], "possible_actions": ans["possible_actions"]}
        evidence = [{"source_id": e["source_id"], "header": e["header"], "text": e["text"]} for e in resp["evidence"]["items"]]
        items.append({"body": body, "evidence": evidence})
        key.append({"model": model, "rep": rep, "case_id": r["case_id"], "arm": r["arm"]})
    order = list(range(len(items)))
    random.Random(20261119).shuffle(order)
    with (D / "packets.jsonl").open("w") as p, (D / "_key.jsonl").open("w") as k:
        for n, i in enumerate(order):
            pid = f"P{n:02d}"
            p.write(json.dumps({"pid": pid, **items[i]}, ensure_ascii=False) + "\n")
            k.write(json.dumps({"pid": pid, **key[i]}, ensure_ascii=False) + "\n")
    print(len(items), "answers packed")


if __name__ == "__main__":
    {"packets": packets, "report": report}[sys.argv[1]]()
