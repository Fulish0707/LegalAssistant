"""Blind labelling packets for the turn replay (PROTOCOL.md).

  runs/<stem>-condense.md       one item per input: the base condenser output, labelled Q / QA / A
  runs/<stem>-pairs-<n>.md      one item per (input, compared arm): the base answer and the base2
                                or message answer as 답변 1 and 답변 2, sides drawn at random; labelled 1 / 2 / =
  runs/<stem>-key.json          which side is which arm; never given to the labellers

Usage: python packet.py <replay .jsonl>
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261125
COMPARED = ("base2", "message")
CHUNK = 30


def render(row: dict) -> str:
    answer = row["answer"]
    if row["response_kind"] != "answer" or answer is None:
        return f"(답변이 나오지 않음: {row['response_kind']})"
    parts = [f"요약: {answer['summary']}", f"분석: {answer['analysis']}"]
    if answer["possible_actions"]:
        parts.append("할 수 있는 일:\n" + "\n".join(f"- {item}" for item in answer["possible_actions"]))
    if answer["additional_information_needed"]:
        parts.append("더 필요한 정보:\n" + "\n".join(f"- {item}" for item in answer["additional_information_needed"]))
    if answer.get("uncertainty"):
        parts.append(f"불확실성: {answer['uncertainty']}")
    return "\n".join(parts)


def context(row: dict) -> str:
    return "이전 대화:\n" + "\n".join(row["window"]) + f"\n\n현재 질문: {row['message']}"


def main() -> None:
    path = Path(sys.argv[1])
    stem = path.with_suffix("").name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by = {(row["input_id"], row["arm"]): row for row in rows}
    inputs = sorted({row["input_id"] for row in rows})
    rng = random.Random(SEED)
    key: dict = {"condense": {}, "pairs": {}}

    order = inputs[:]
    rng.shuffle(order)
    items = []
    for number, input_id in enumerate(order, 1):
        item_id = f"C{number:03d}"
        row = by[(input_id, "base")]
        items.append(f"## {item_id}\n\n{context(row)}\n\n정리 결과: {row['condensed_raw']}\n")
        key["condense"][item_id] = input_id
    (HERE / "runs" / f"{stem}-condense.md").write_text("\n".join(items))

    pairs = [(input_id, arm) for input_id in inputs for arm in COMPARED]
    rng.shuffle(pairs)
    chunks: list[list[str]] = []
    for number, (input_id, arm) in enumerate(pairs, 1):
        item_id = f"P{number:03d}"
        base, other = by[(input_id, "base")], by[(input_id, arm)]
        base_first = rng.random() < 0.5
        first, second = (base, other) if base_first else (other, base)
        text = f"## {item_id}\n\n{context(base)}\n\n### 답변 1\n{render(first)}\n\n### 답변 2\n{render(second)}\n"
        if (number - 1) % CHUNK == 0:
            chunks.append([])
        chunks[-1].append(text)
        key["pairs"][item_id] = {"input_id": input_id, "arm": arm, "base_side": 1 if base_first else 2}
    for number, chunk in enumerate(chunks, 1):
        (HERE / "runs" / f"{stem}-pairs-{number}.md").write_text("\n".join(chunk))
    (HERE / "runs" / f"{stem}-key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2))
    print(len(items), "condense items,", len(pairs), "pairs in", len(chunks), "packets")


if __name__ == "__main__":
    main()
