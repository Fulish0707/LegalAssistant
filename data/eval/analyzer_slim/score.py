"""Score runs.jsonl against the preregistered gates (PROTOCOL.md). Prints and writes gates.json."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from legal_rag.rag.clarify import ClarificationPolicy
from legal_rag.schemas import QueryAnalysis

HERE = Path(__file__).parent
rows = [json.loads(line) for line in (HERE / "runs.jsonl").read_text().splitlines()]
by = defaultdict(dict)
for r in rows:
    by[r["qid"]][r["arm"]] = r
policy = ClarificationPolicy(allow=True)


def clarify(r):
    return policy.should_clarify(QueryAnalysis.model_validate(r["analysis"]))


def refs(r):
    a = r["analysis"]
    return (tuple(a["statute_refs"]), tuple(a["case_refs"]))


def open_q(r):
    a = r["analysis"]
    return bool(a["needs_more_info"] and a["missing_facts"])


def failed(r):
    return bool(r.get("error")) or not r.get("parsed", False)


def agree(pairs, f):
    return sum(f(a) == f(b) for a, b in pairs) / len(pairs)


pairs = [(v["full"], v["slim"]) for v in by.values() if "full" in v and "slim" in v]
rep = [(v["full"], v["full2"]) for v in by.values() if "full" in v and "full2" in v]
med_full = statistics.median(a["wall_s"] for a, _ in pairs)
med_slim = statistics.median(b["wall_s"] for _, b in pairs)
out = {
    "n": len(pairs),
    "G1_clarify_agreement": agree(pairs, clarify),
    "G1_clarify_count": {"full": sum(clarify(a) for a, _ in pairs), "slim": sum(clarify(b) for _, b in pairs)},
    "G2_refs_agreement": agree(pairs, refs),
    "G3_open_questions_agreement": agree(pairs, open_q),
    "open_questions_count": {"full": sum(open_q(a) for a, _ in pairs), "slim": sum(open_q(b) for _, b in pairs)},
    "G4_median_wall_s": {"full": med_full, "slim": med_slim, "reduction": 1 - med_slim / med_full},
    "p90_wall_s": {k: sorted(x[i]["wall_s"] for x in pairs)[int(len(pairs) * 0.9)] for i, k in ((0, "full"), (1, "slim"))},
    "median_out_tokens": {"full": statistics.median(a["out_tokens"] or 0 for a, _ in pairs), "slim": statistics.median(b["out_tokens"] or 0 for _, b in pairs)},
    "G5_failures": {"full": sum(failed(a) for a, _ in pairs), "slim": sum(failed(b) for _, b in pairs)},
    "domain_agreement": agree(pairs, lambda r: r["analysis"]["domain"]),
    "reference_full_full": {"n": len(rep), "clarify": agree(rep, clarify) if rep else None, "refs": agree(rep, refs) if rep else None,
                            "open_questions": agree(rep, open_q) if rep else None, "domain": agree(rep, lambda r: r["analysis"]["domain"]) if rep else None},
}
c, g = out["G1_clarify_count"], out
out["gates"] = {
    "G1": g["G1_clarify_agreement"] >= 0.97 and abs(c["full"] - c["slim"]) <= 3,
    "G2": g["G2_refs_agreement"] >= 0.97,
    "G3": g["G3_open_questions_agreement"] >= 0.90,
    "G4": g["G4_median_wall_s"]["reduction"] >= 0.40,
    "G5": g["G5_failures"]["slim"] <= g["G5_failures"]["full"],
}
out["adopt"] = all(out["gates"].values())
out["disagreements"] = {
    "clarify": [k for k, v in by.items() if "slim" in v and "full" in v and clarify(v["full"]) != clarify(v["slim"])],
    "refs": [k for k, v in by.items() if "slim" in v and "full" in v and refs(v["full"]) != refs(v["slim"])],
    "open_questions": [k for k, v in by.items() if "slim" in v and "full" in v and open_q(v["full"]) != open_q(v["slim"])],
}
print(json.dumps(out, ensure_ascii=False, indent=1))
(HERE / "gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
