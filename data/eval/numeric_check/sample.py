"""Eval sample, blind batches and scoring for the quantity relation detector (PROTOCOL.md).

  python sample.py build                 # needs _key/rule.FROZEN; writes _key/sample.jsonl and batches/
  python sample.py score labels-dir      # merges labels, reports agreement, P, R, cost (needs adjudication.jsonl if any disagreement)
"""
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from detect import ROOT, eval_rows  # noqa: E402
from legal_rag.verification.quantities import _sentence, check_quantities, extract_quantities  # noqa: E402

SEED = 20261123
MAX_FLAGGED = 60
UNFLAGGED = 60
ERRORS = {"틀림-관계", "틀림-적용", "틀림-출처없음"}
LABELS = ERRORS | {"맞음", "unclear"}


def frozen_ok() -> None:
    frozen = json.loads((HERE / "_key/rule.FROZEN").read_text())
    actual = hashlib.sha256((ROOT / frozen["rule"]).read_bytes()).hexdigest()
    assert actual == frozen["sha256"], "rule changed after freezing"


def mark(answer: str, sentence: str) -> str:
    i = answer.find(sentence)
    return answer if i < 0 else answer[:i] + "⟦" + sentence + "⟧" + answer[i + len(sentence):]


def build() -> None:
    frozen_ok()
    units, answers, flagged_answers = [], 0, 0
    for row in eval_rows():
        answers += 1
        findings = check_quantities(row["answer"], row["evidence"], row["user"])
        flagged_answers += bool(findings)
        flagged = defaultdict(list)
        for f in findings:
            flagged[f.sentence].append(f.rule)
        sentences = {_sentence(row["answer"], q) for q in extract_quantities(row["answer"])}
        for s in sorted(sentences):
            if s:
                units.append({"answer_id": row["id"], "sentence": s, "flagged": s in flagged, "rules": sorted(set(flagged.get(s, []))),
                              "question": row["user"], "evidence": row["evidence"], "answer": row["answer"]})
    rng = random.Random(SEED)
    pos = [u for u in units if u["flagged"]]
    neg = [u for u in units if not u["flagged"]]
    take_pos = pos if len(pos) <= MAX_FLAGGED else rng.sample(pos, MAX_FLAGGED)
    take_neg = rng.sample(neg, min(UNFLAGGED, len(neg)))
    sample = take_pos + take_neg
    rng.shuffle(sample)
    key_dir = HERE / "_key"
    with (key_dir / "sample.jsonl").open("w") as fh:
        for i, u in enumerate(sample):
            u["item_id"] = f"n{i:03d}"
            fh.write(json.dumps({k: u[k] for k in ("item_id", "answer_id", "sentence", "flagged", "rules")}, ensure_ascii=False) + "\n")
    (key_dir / "population.json").write_text(json.dumps({
        "answers": answers, "flagged_answers": flagged_answers, "units": len(units), "flagged_units": len(pos),
        "unflagged_units": len(neg), "sampled_flagged": len(take_pos), "sampled_unflagged": len(take_neg)}, indent=2) + "\n")
    half = math.ceil(len(sample) / 2)
    for name, chunk in (("A", sample[:half]), ("B", sample[half:])):
        for labeler, order in (("1", chunk), ("2", list(reversed(chunk)))):
            d = HERE / "batches" / f"{name}{labeler}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "GUIDELINES.md").write_text((HERE / "GUIDELINES.md").read_text())
            with (d / "batch.jsonl").open("w") as fh:
                for u in order:
                    fh.write(json.dumps({"item_id": u["item_id"], "question": u["question"], "evidence": u["evidence"],
                                         "answer": mark(u["answer"], u["sentence"])}, ensure_ascii=False) + "\n")
    print(json.loads((key_dir / "population.json").read_text()))


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_labels(directory: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(directory.iterdir()):
        path = d / "labels.jsonl"
        if not path.exists():
            continue
        expected = {json.loads(line)["item_id"] for line in (d / "batch.jsonl").read_text().splitlines()}
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert {r["item_id"] for r in rows} == expected and len(rows) == len(expected), d
        assert all(r["label"] in LABELS for r in rows), d
        for r in rows:
            out[r["item_id"]][d.name] = r
    return out


def score(directory: Path) -> None:
    frozen_ok()
    key = {json.loads(line)["item_id"]: json.loads(line) for line in (HERE / "_key/sample.jsonl").read_text().splitlines()}
    population = json.loads((HERE / "_key/population.json").read_text())
    labels = load_labels(directory)
    assert set(labels) == set(key), "every sampled item needs labels"
    adjudicated = {}
    adj_path = HERE / "adjudication.jsonl"
    if adj_path.exists():
        adjudicated = {json.loads(line)["item_id"]: json.loads(line) for line in adj_path.read_text().splitlines()}
    pairs, final, disagreements = [], {}, []
    for item, by in labels.items():
        a, b = [by[k]["label"] for k in sorted(by)]
        if a != "unclear" and b != "unclear":
            pairs.append((a in ERRORS, b in ERRORS))
        if (a in ERRORS) == (b in ERRORS) and "unclear" not in (a, b):
            final[item] = a if a in ERRORS else "맞음"
        elif item in adjudicated:
            final[item] = adjudicated[item]["label"]
        else:
            disagreements.append(item)
    print(f"two-class agreement kappa {kappa(pairs):.2f} on {len(pairs)} items; disagreements {len(disagreements)}")
    if disagreements:
        print("adjudicate first (adjudication.jsonl):", disagreements)
        return
    for strict in (True, False):
        is_error = (lambda lab: lab in ERRORS) if strict else (lambda lab: lab in ERRORS or lab == "unclear")
        pos = [i for i in key if key[i]["flagged"] and final[i] != "unclear" or (not strict and key[i]["flagged"])]
        neg = [i for i in key if not key[i]["flagged"] and (final[i] != "unclear" or not strict)]
        tp = sum(is_error(final[i]) for i in pos)
        neg_err = sum(is_error(final[i]) for i in neg)
        precision = tp / len(pos) if pos else 0.0
        neg_rate = neg_err / len(neg) if neg else 0.0
        est_missed = neg_rate * population["unflagged_units"]
        flagged_errors = precision * population["flagged_units"]
        recall = flagged_errors / (flagged_errors + est_missed) if flagged_errors + est_missed else 0.0
        lo, hi = wilson(tp, len(pos))
        cost = population["flagged_answers"] / population["answers"]
        name = "main (unclear excluded)" if strict else "sensitivity (unclear = error)"
        print(f"\n## {name}\n- P {tp}/{len(pos)} = {precision:.2f} (Wilson {lo:.2f}-{hi:.2f})"
              f"\n- unflagged error rate {neg_err}/{len(neg)} = {neg_rate:.2f} -> est. missed {est_missed:.1f} of {population['unflagged_units']}"
              f"\n- R (est.) {recall:.2f}\n- flagged answers {population['flagged_answers']}/{population['answers']} = {cost:.1%}")
        if strict:
            enough = len(pos) >= 10
            gates = {"P>=0.6 & Wilson lower>=0.4": enough and precision >= 0.6 and lo >= 0.4, "R>=0.3": recall >= 0.3, "cost<=15%": cost <= 0.15}
            print("- gates:", gates if enough else "fewer than 10 flagged sample items: not judged")
    by_rule = Counter()
    for i in key:
        for rule in key[i]["rules"]:
            by_rule[(rule, final[i] in ERRORS)] += 1
    print("\nper rule (rule, is_error): count", dict(by_rule))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        score(Path(sys.argv[2]))
