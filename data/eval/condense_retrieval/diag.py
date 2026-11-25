"""Does the condenser's defect change retrieval? Diagnostic (PLAN.md). Retrieval only, no model call.

Usage (serving .env loaded, cwd = repo root):
  ( set -a; . ./.env; set +a; exec env -u PYTHONPATH -u PYTHONHOME .venv/bin/python data/eval/condense_retrieval/diag.py )
Writes runs/<tag>-diag.jsonl and prints the summary of PLAN.md.
"""
import json
import time
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator

HERE = Path(__file__).parent
EVAL = HERE.parent
RANK = {"Q": 0, "QA": 1, "A": 2}


def worse(a, b):
    return max(a, b, key=lambda mark: RANK[mark])


def summary_labels() -> dict[str, str]:
    run = EVAL / "condense_summary" / "runs" / "20261124T170228-replay"
    key = json.loads(Path(f"{run}-key.json").read_text())
    a, b = (json.loads(Path(f"{run}-labels_{n}.json").read_text()) for n in (1, 2))
    labels: dict[str, str] = {}
    for item, entry in key.items():
        if any(use["arm"] == "base" for use in entry["uses"]):
            mark = worse(a[item], b[item])
            labels[entry["input_id"]] = worse(labels.get(entry["input_id"], "Q"), mark)
    return labels


def downstream_labels() -> dict[str, str]:
    run = EVAL / "condense_downstream" / "runs" / "20261124T234341-replay"
    key = json.loads(Path(f"{run}-key.json").read_text())["condense"]
    a, b = (json.loads(Path(f"{run}-condense_labels_{n}.json").read_text()) for n in (1, 2))
    return {input_id: worse(a[item], b[item]) for item, input_id in key.items()}


SETS = [
    ("summary", EVAL / "condense_summary" / "runs" / "20261124T162734-base-scenarios_heldout.jsonl",
     EVAL / "condense_summary" / "scenarios_heldout.json", summary_labels),
    ("downstream", EVAL / "condense_downstream" / "runs" / "20261124T230324-base-scenarios_heldout.jsonl",
     EVAL / "condense_downstream" / "scenarios_heldout.json", downstream_labels),
]


def inputs():
    for name, record, scenarios, labeller in SETS:
        expected = {s["id"]: s["evidence_any"] for s in json.loads(scenarios.read_text())}
        labels = labeller()
        for line in record.read_text().splitlines():
            row = json.loads(line)
            for index, turn in enumerate(row["turns"]):
                if turn.get("condensed_raw") is None or not turn.get("condense_input"):
                    continue
                input_id = f"{row['id']}#{index + 1}"
                history = turn["condense_input"]["history"][-4:]
                users = [t["content"] for t in history if t["role"] == "user"] + [turn["message"]]
                yield {"set": name, "input_id": input_id, "label": labels[input_id],
                       "evidence_any": expected[row["id"]], "message": turn["message"],
                       "served": turn["search_queries"], "joined": " ".join(u.strip() for u in users if u.strip())}


def main() -> None:
    settings = load_settings()
    orch = build_orchestrator(settings)
    k = settings.retrieval.k_final
    tag = time.strftime("%Y%m%dT%H%M%S")
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    rows = []
    with (out / f"{tag}-diag.jsonl").open("w") as fh:
        for item in inputs():
            row = {key: item[key] for key in ("set", "input_id", "label", "message", "joined", "served", "evidence_any")}
            for arm in ("served", "joined", "message"):
                first = item["served"][0] if arm == "served" else item[arm]
                queries = [first, *item["served"][1:]]
                hits = orch.retriever.retrieve_many(queries, k, None, rerank_query=first)
                bundle = orch.context_builder.build(hits)
                text = "\n".join(json.dumps(e.header, ensure_ascii=False) + "\n" + e.text for e in bundle.items)
                row[arm] = {"ids": [e.chunk_id for e in bundle.items],
                            "hit": any(word in text for word in item["evidence_any"])}
            a, b = set(row["served"]["ids"]), set(row["joined"]["ids"])
            row["jaccard"] = round(len(a & b) / len(a | b), 3) if a | b else 1.0
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(row["input_id"], row["label"], row["served"]["hit"], row["joined"]["hit"], row["message"]["hit"], row["jaccard"], flush=True)
    report: dict = {"tag": tag}
    for group, members in (("D", [r for r in rows if r["label"] != "Q"]), ("G", [r for r in rows if r["label"] == "Q"])):
        report[group] = {
            "n": len(members),
            "hits": {arm: sum(r[arm]["hit"] for r in members) for arm in ("served", "joined", "message")},
            "joined_only": sorted(r["input_id"] for r in members if r["joined"]["hit"] and not r["served"]["hit"]),
            "served_only": sorted(r["input_id"] for r in members if r["served"]["hit"] and not r["joined"]["hit"]),
            "mean_jaccard": round(sum(r["jaccard"] for r in members) / len(members), 3) if members else None,
        }
        report[group]["net_gain_joined"] = len(report[group]["joined_only"]) - len(report[group]["served_only"])
    (out / f"{tag}-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
