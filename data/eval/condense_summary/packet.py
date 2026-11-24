"""Blind labelling packet for the replay layer (PROTOCOL.md).

Every distinct raw condenser output per input (across arms and repeats) becomes one item with a
random id. The packet shows the conversation (the stored turns of the condenser's window, in
full, as the base arm received them before its 500-character cut) and the output; it never shows the arm. The key that maps items back to arms is
written separately and is not given to the labellers.

Usage: python packet.py <replay .jsonl>   -> runs/<stem>-packet.md, runs/<stem>-key.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261124


def main() -> None:
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # The stored window in full (replay.py "window"); both arms share the input.
    context = {row["input_id"]: (row["message"], row["window"]) for row in rows}
    outputs: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["raw"] is None:
            continue
        outputs.setdefault((row["input_id"], row["raw"].strip()), []).append({"arm": row["arm"], "repeat": row["repeat"]})
    keys = list(outputs)
    random.Random(SEED).shuffle(keys)
    items, key = [], {}
    for number, (input_id, text) in enumerate(keys, 1):
        item_id = f"L{number:03d}"
        message, turns = context[input_id]
        items.append(f"## {item_id}\n\n이전 대화:\n" + "\n".join(turns) + f"\n\n현재 질문: {message}\n\n정리 결과: {text}\n")
        key[item_id] = {"input_id": input_id, "text": text, "uses": outputs[(input_id, text)]}
    stem = path.with_suffix("").name
    (HERE / "runs" / f"{stem}-packet.md").write_text("\n".join(items))
    (HERE / "runs" / f"{stem}-key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2))
    print(len(items), "items")


if __name__ == "__main__":
    main()
